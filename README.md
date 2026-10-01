# Skyline Residency — AI voice sales agent (demo)

Talk to **Riya**, a Hinglish-speaking real-estate sales assistant, by voice in real time,
with natural interruptions. Native speech-to-speech with **Gemini 3.8 Live** through
**Google ADK** bidi streaming. No separate STT/TTS.

```
Browser (mic/speaker) ─┐                                   
                       ├─ WebSocket ─ FastAPI bridge ─ ADK run_live ─ gemini-3.8-live
Twilio phone call ─────┘   /ws/web  /ws/twilio     (LiveRequestQueue)
```

- The API key stays on the server. One ADK live session per call, closed when the socket closes.
- Barge-in: when Gemini reports `interrupted`, the server tells the browser to flush its
  playback queue, or sends Twilio a `clear` event.
- Each call's transcript is saved to `calls/<timestamp>-<web|twilio>.json`.
- Site visit bookings are appended to `bookings.json`.

| File | What it does |
|---|---|
| `app.py` | FastAPI: serves the page, `/ws/web`, `/twilio/voice`, `/ws/twilio` |
| `agent.py` | Riya's prompt, tools, ADK runner, and `LiveCall` (shared by both transports) |
| `audio.py` | mu-law ⇄ PCM16 and streaming 8k⇄16k/24k resampling (numpy) |
| `project_data.py` | The only project facts Riya is allowed to quote |
| `static/index.html` | Call button, status, live transcript, playback with jitter buffer |
| `static/audio-worklet.js` | Mic capture → PCM16 mono 16 kHz |

Audio formats: Gemini Live input is PCM16 mono **16 kHz**, output is PCM16 mono **24 kHz**.
Twilio is base64 mu-law **8 kHz**.

## Setup

1. **Get a Gemini API key** (free tier works): https://aistudio.google.com/apikey
2. **Install** (Python 3.11+):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env      # then put your key in GOOGLE_API_KEY
   ```
3. **Run**:
   ```bash
   uvicorn app:app --port 8000
   ```
   On a laptop, open http://localhost:8000 (localhost counts as secure, so the mic works).

## Phase 1: call from your phone's browser

Phone browsers only allow the mic over HTTPS, so expose the server with ngrok:

```bash
ngrok http 8000
```

Open the `https://….ngrok-free.app` URL on your phone (tap "Visit Site" on ngrok's
free-tier warning page), tap **Call**, allow the mic, and start talking. Riya greets
you first. Talk over her to interrupt.

> Tip: if Riya keeps interrupting herself on speakerphone, the mic is hearing her.
> Use earphones, or turn the volume down.

## Phase 2: inbound phone calls via Twilio (trial)

1. Sign up at https://www.twilio.com/try-twilio and get a free trial phone number.
2. Under **Phone Numbers → Verified Caller IDs**, verify the phone you will call from.
   Trial numbers only accept calls from verified numbers.
3. Keep `uvicorn` and `ngrok http 8000` running. Under **Phone Numbers → Manage → Active
   numbers → your number → Voice configuration**, set **A call comes in** to
   *Webhook* `https://<your-ngrok-host>/twilio/voice`, method **HTTP POST**. Save.
4. Call the number from your verified phone. Trial accounts first play a short Twilio
   notice and ask you to press a key. After that you are talking to Riya.

`/twilio/voice` returns `<Connect><Stream url="wss://<host>/ws/twilio"/></Connect>`, taking
the host from the request. If that comes out wrong behind a proxy, set
`PUBLIC_HOST=<your-ngrok-host>` in `.env`.

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. AI Studio key. |
| `GOOGLE_GENAI_USE_VERTEXAI` | `FALSE` | Keep `FALSE` for AI Studio keys. |
| `GEMINI_MODEL` | `gemini-3.8-live` | `gemini-3.8-live-extended-thinking` also works, with more reasoning and slower replies. |
| `GEMINI_VOICE` | `Kore` | A Gemini prebuilt voice name. Leave it empty to use the model's default voice. |
| `INLINE_PROJECT_INFO` | `0` | `1` puts the project facts in the system prompt instead of the `get_project_info` tool. Use it if tool calls misbehave or add noticeable delay. Booking still uses the tool. |
| `PUBLIC_HOST` | auto | Host used in the Twilio stream URL. |

## The agent

Riya speaks Hinglish by default, and switches to English or Hindi if you do. She keeps
replies to one or two spoken sentences. Her flow is: greet → ask what you're looking
for → qualify (budget, 2/3 BHK, location, timeline, ready vs under-construction) →
answer questions → offer a site visit → close.

Tools (plain Python functions registered with ADK, marked `BLOCKING` so the model waits
for real facts rather than improvising):
- `get_project_info()` returns the dict in `project_data.py`.
- `book_site_visit(name, phone, preferred_date, preferred_time)` appends to `bookings.json`.

To change the project, edit `project_data.py`. To change her personality or flow, edit
`INSTRUCTION` in `agent.py`.

## Troubleshooting

- **The status shows "Error: … API key not valid"**: check `GOOGLE_API_KEY` in `.env`.
- **No sound on iPhone**: make sure the ringer/silent switch is off. The page has to be
  opened over HTTPS (ngrok).
- **Long pause before the first answer**: try `INLINE_PROJECT_INFO=1` to skip the tool round trip.
- **Calls stop after a few minutes**: Gemini Live connections last about 10 minutes.
  Session resumption is enabled, so ADK reconnects transparently when the server sends GoAway.
