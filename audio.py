"""Audio conversions for the Twilio leg (numpy only; audioop is gone in 3.13).

Twilio Media Streams: base64 G.711 mu-law, 8 kHz mono.
Gemini Live:          PCM16 little-endian mono, 16 kHz in / 24 kHz out.
"""

import numpy as np


# ---------- G.711 mu-law ----------

_BIAS = 0x84
_CLIP = 8159  # 14-bit, as in G.711 / audioop


def _build_ulaw_decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign, -sample, sample).astype(np.int16)


_ULAW_DECODE = _build_ulaw_decode_table()


def ulaw_to_pcm16(data: bytes) -> bytes:
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)].tobytes()


def pcm16_to_ulaw(data: bytes) -> bytes:
    # Same 14-bit algorithm as CPython's audioop.lin2ulaw (bit-exact).
    x = np.frombuffer(data, dtype="<i2").astype(np.int32) >> 2
    mask = np.where(x < 0, 0x7F, 0xFF)
    x = np.minimum(np.abs(x), _CLIP) + (_BIAS >> 2)
    seg = np.clip(np.floor(np.log2(x)).astype(np.int32) - 5, 0, 7)
    uval = np.where(x > 0x1FFF, 0x7F, (seg << 4) | ((x >> (seg + 1)) & 0x0F))
    return (uval ^ mask).astype(np.uint8).tobytes()


# ---------- streaming resampler ----------

class Resampler:
    """Streaming PCM16 resampler by a rational factor up/down.

    Zero-stuff by `up`, low-pass with a windowed-sinc FIR, keep every `down`-th
    sample. Filter state and decimation phase carry across chunks, so there are
    no clicks at chunk boundaries.
    """

    def __init__(self, up: int, down: int, taps: int = 63):
        self.up, self.down = up, down
        cutoff = 0.5 / max(up, down)  # cycles/sample at the upsampled rate
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cutoff * n) * np.hamming(taps)
        self.h = (h / h.sum() * up).astype(np.float32)
        self.tail = np.zeros(taps - 1, dtype=np.float32)
        self.pos = 0  # running sample index (mod down) at the upsampled rate

    def process(self, data: bytes) -> bytes:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32)
        if self.up > 1:
            z = np.zeros(len(x) * self.up, dtype=np.float32)
            z[:: self.up] = x
            x = z
        buf = np.concatenate([self.tail, x])
        y = np.convolve(buf, self.h, mode="valid")  # len(y) == len(x)
        self.tail = buf[len(buf) - len(self.tail):]
        offset = (-self.pos) % self.down
        self.pos = (self.pos + len(x)) % self.down
        y = y[offset:: self.down]
        return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


def twilio_in() -> Resampler:
    """mu-law-decoded 8 kHz -> 16 kHz for Gemini."""
    return Resampler(up=2, down=1)


def twilio_out() -> Resampler:
    """Gemini 24 kHz -> 8 kHz for Twilio."""
    return Resampler(up=1, down=3)
