"""FastAPI bridge: browser / Twilio <-> WebSocket <-> ADK live session <-> Gemini Live."""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before importing agent, which reads env vars

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import audio  # noqa: E402
from agent import LiveCall  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")

STATIC = Path(__file__).parent / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


async def bridge(call: LiveCall, upstream, on_event) -> Exception | None:
    """Run caller->model (upstream) and model->caller (on_event) until either ends.
    Returns the error that ended the call, if any."""
    async def downstream():
        async for event in call.events():
            await on_event(event)

    tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
    error = None
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                error = t.exception()
                log.error("Call error", exc_info=error)
    finally:
        call.close()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return error


# ---------- Phase 1: browser ----------

@app.websocket("/ws/web")
async def ws_web(ws: WebSocket):
    """Binary frames in: PCM16 16 kHz. Binary frames out: PCM16 24 kHz.
    Text frames out: JSON {"type": "transcript" | "interrupted" | ...}."""
    await ws.accept()
    call = LiveCall("web")
    await call.start()
    log.info("Web call started")

    async def upstream():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                call.send_audio(msg["bytes"])

    async def on_event(ev):
        if ev["type"] == "audio":
            await ws.send_bytes(ev["data"])
        else:
            await ws.send_text(json.dumps(ev, ensure_ascii=False))

    error = await bridge(call, upstream, on_event)
    log.info("Web call ended")
    try:
        if error:
            await ws.send_text(json.dumps({"type": "error", "message": str(error)}))
        await ws.close()
    except RuntimeError:
        pass  # already closed by the browser


# ---------- Phase 2: Twilio ----------

def _public_host(request_headers) -> str:
    return (os.getenv("PUBLIC_HOST")
            or request_headers.get("x-forwarded-host")
            or request_headers["host"])


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    host = _public_host(request.headers)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{host}/ws/twilio"/></Connect></Response>'
    )
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws/twilio")
async def ws_twilio(ws: WebSocket):
    """Twilio Media Streams: JSON text frames, audio is base64 mu-law 8 kHz."""
    await ws.accept()

    # Wait for the "start" event to learn the streamSid.
    stream_sid = None
    while stream_sid is None:
        msg = json.loads(await ws.receive_text())
        if msg["event"] == "start":
            stream_sid = msg["start"]["streamSid"]
        elif msg["event"] == "stop":
            return

    call = LiveCall("twilio")
    await call.start()
    log.info("Twilio call started: %s", stream_sid)
    up_rs, down_rs = audio.twilio_in(), audio.twilio_out()

    async def upstream():
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["event"] == "media":
                ulaw = base64.b64decode(msg["media"]["payload"])
                call.send_audio(up_rs.process(audio.ulaw_to_pcm16(ulaw)))
            elif msg["event"] == "stop":
                return

    async def on_event(ev):
        if ev["type"] == "audio":
            ulaw = audio.pcm16_to_ulaw(down_rs.process(ev["data"]))
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(ulaw).decode()},
            }))
        elif ev["type"] == "interrupted":
            await ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
        elif ev["type"] == "transcript" and ev["final"]:
            log.info("[%s] %s", ev["role"], ev["text"])

    await bridge(call, upstream, on_event)
    log.info("Twilio call ended: %s", stream_sid)
    try:
        await ws.close()
    except RuntimeError:
        pass
