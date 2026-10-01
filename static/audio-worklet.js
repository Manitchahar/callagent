// Mic capture: downsample the context's native rate (usually 48 kHz) to 16 kHz
// PCM16 mono and post ~40 ms chunks to the main thread.
const TARGET_RATE = 16000;
const CHUNK = 640; // 40 ms at 16 kHz

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.out = new Int16Array(CHUNK);
    this.n = 0;
    this.acc = 0;   // sum of input samples in the current output window
    this.cnt = 0;   // how many input samples are in it
    this.pos = 0;   // input samples consumed toward the next output sample
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // Box-filter average over each output window (cheap anti-aliasing).
      this.acc += ch[i];
      this.cnt++;
      this.pos++;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const s = Math.max(-1, Math.min(1, this.acc / this.cnt));
        this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
        this.acc = 0;
        this.cnt = 0;
        if (this.n === CHUNK) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(CHUNK);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
