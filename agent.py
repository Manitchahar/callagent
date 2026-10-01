"""Riya, the Skyline Residency voice agent, and the transport-agnostic live call.

Both the browser (/ws/web) and Twilio (/ws/twilio) use LiveCall; they only
differ in how audio gets in and out.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from google.adk.agents import Agent, LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from project_data import PROJECT_INFO

log = logging.getLogger("agent")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-live")
VOICE = os.getenv("GEMINI_VOICE", "Kore")
# Fallback if tool calls misbehave: put project facts in the prompt instead.
INLINE_PROJECT_INFO = os.getenv("INLINE_PROJECT_INFO", "0") == "1"

INPUT_RATE = 16000  # Gemini Live input: PCM16 mono 16 kHz
OUTPUT_RATE = 24000  # Gemini Live output: PCM16 mono 24 kHz

BASE_DIR = Path(__file__).parent
CALLS_DIR = BASE_DIR / "calls"
BOOKINGS_FILE = BASE_DIR / "bookings.json"


# ---------- tools ----------

def get_project_info() -> dict:
    """Returns all known facts about Skyline Residency: location, configurations,
    prices, amenities, possession dates, payment plans and RERA number.
    Call this before answering any question about the project."""
    return PROJECT_INFO


def book_site_visit(name: str, phone: str, preferred_date: str, preferred_time: str) -> dict:
    """Books a site visit at Skyline Residency.

    Args:
        name: Caller's full name.
        phone: Caller's 10-digit mobile number.
        preferred_date: Preferred visit date, e.g. "Saturday 12 October".
        preferred_time: Preferred visit time, e.g. "11 AM".
    """
    bookings = json.loads(BOOKINGS_FILE.read_text()) if BOOKINGS_FILE.exists() else []
    booking = {
        "id": uuid.uuid4().hex[:6].upper(),
        "name": name,
        "phone": phone,
        "preferred_date": preferred_date,
        "preferred_time": preferred_time,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    bookings.append(booking)
    BOOKINGS_FILE.write_text(json.dumps(bookings, indent=2, ensure_ascii=False))
    log.info("Site visit booked: %s", booking)
    return {
        "status": "confirmed",
        "booking_id": booking["id"],
        "message": f"Site visit booked for {name} on {preferred_date} at {preferred_time}. "
                   "The sales team will call to confirm.",
    }


# ---------- agent ----------

INSTRUCTION = """
You are Riya, a warm and professional voice sales assistant for Skyline Residency,
a residential project. You are on a live voice call.

Language: speak natural Hinglish (Hindi and English mixed, the way people talk in
Delhi or Mumbai) by default. If the caller speaks only English, reply in English.
If the caller speaks pure Hindi, reply in Hindi.

Style: this is a phone call. Keep every reply to one or two short spoken sentences.
Never use lists, bullet points, markdown, or emojis. Ask one question at a time.
Say prices the way people speak them, like "eighty-five lakh" or "one point two five crore".

Call flow:
1. Greet the caller, introduce yourself as Riya from Skyline Residency, ask their name.
2. Ask what they are looking for.
3. Qualify naturally over a few turns: budget, 2 BHK or 3 BHK, preferred location,
   timeline to buy, and whether they want ready-to-move or under-construction.
4. Answer their questions.
5. Offer a site visit. If they agree, collect name, mobile number, preferred date and
   time, repeat them back once, then book it with the book_site_visit tool.
6. Thank them and close politely.

Facts: {facts_rule}
Never invent prices, dates, offers, discounts, or any other detail. If something is
not in the project info, say the sales team will confirm it.
"""

if INLINE_PROJECT_INFO:
    _facts_rule = ("Use only this project info:\n"
                   + json.dumps(PROJECT_INFO, indent=1, ensure_ascii=False))
    _tools = [book_site_visit]
else:
    _facts_rule = ("Before stating any project fact, call get_project_info and use "
                   "only what it returns.")
    # BLOCKING: the model waits for the facts instead of improvising meanwhile.
    _info_tool = FunctionTool(get_project_info)
    _info_tool.behavior = types.Behavior.BLOCKING
    _book_tool = FunctionTool(book_site_visit)
    _book_tool.behavior = types.Behavior.BLOCKING
    _tools = [_info_tool, _book_tool]

root_agent = Agent(
    name="riya",
    model=MODEL,
    description="Voice sales assistant for Skyline Residency.",
    instruction=INSTRUCTION.format(facts_rule=_facts_rule),
    tools=_tools,
)

APP_NAME = "skyline"
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)


def _run_config() -> RunConfig:
    return RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ) if VOICE else None,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # Lets ADK reconnect transparently when the server sends GoAway (~10 min).
        session_resumption=types.SessionResumptionConfig(),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
    )


# ---------- one call ----------

class LiveCall:
    """One phone/web call = one ADK live session.

    Feed caller audio with send_audio(); iterate events() for what to play/show.
    events() yields dicts:
      {"type": "audio", "data": <PCM16 24 kHz bytes>}
      {"type": "interrupted"}
      {"type": "transcript", "role": "user"|"agent", "text": str, "final": bool}
    """

    def __init__(self, channel: str):
        self.channel = channel
        self.user_id = f"{channel}-caller"
        self.started_at = datetime.now()
        self.queue = LiveRequestQueue()
        self.transcript: list[dict] = []
        self._closed = False

    async def start(self):
        session = await runner.session_service.create_session(
            app_name=APP_NAME, user_id=self.user_id
        )
        self.session_id = session.id
        # Kick off the conversation so Riya greets first.
        self.queue.send_content(types.Content(
            role="user",
            parts=[types.Part(text="(The caller has just connected. Greet them now.)")],
        ))

    def send_audio(self, pcm16_16k: bytes):
        if not self._closed:
            self.queue.send_realtime(
                types.Blob(data=pcm16_16k, mime_type=f"audio/pcm;rate={INPUT_RATE}")
            )

    async def events(self):
        async for event in runner.run_live(
            user_id=self.user_id,
            session_id=self.session_id,
            live_request_queue=self.queue,
            run_config=_run_config(),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.inline_data and part.inline_data.data and \
                            (part.inline_data.mime_type or "").startswith("audio/"):
                        yield {"type": "audio", "data": part.inline_data.data}
                    elif part.function_call:
                        log.info("Tool call: %s(%s)", part.function_call.name,
                                 part.function_call.args)

            if event.interrupted:
                yield {"type": "interrupted"}

            for role, tr in (("user", event.input_transcription),
                             ("agent", event.output_transcription)):
                if tr and tr.text:
                    final = bool(tr.finished)
                    if final:
                        self.transcript.append({
                            "role": role, "text": tr.text.strip(),
                            "at": datetime.now().isoformat(timespec="seconds"),
                        })
                    yield {"type": "transcript", "role": role, "text": tr.text, "final": final}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.queue.close()
        CALLS_DIR.mkdir(exist_ok=True)
        path = CALLS_DIR / f"{self.started_at:%Y%m%d-%H%M%S}-{self.channel}.json"
        path.write_text(json.dumps({
            "channel": self.channel,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "transcript": self.transcript,
        }, indent=2, ensure_ascii=False))
        log.info("Saved transcript to %s", path)
