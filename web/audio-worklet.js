// Streams the microphone as 16 kHz 16-bit PCM in 32 ms frames while active.
class PCMInput extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = false;
    this.buffer = new Int16Array(512);
    this.offset = 0; this.phase = 0; this.sum = 0; this.count = 0;
    this.port.onmessage = ({data}) => {
      this.active = data === 'start';
      this.offset = 0; this.phase = 0; this.sum = 0; this.count = 0;
    };
  }
  process(inputs) {
    if (!this.active || !inputs[0]?.[0]) return true;
    // Average samples when the device cannot run at the requested 16 kHz.
    for (const value of inputs[0][0]) {
      this.sum += value; this.count++;
      this.phase += 16000;
      if (this.phase < sampleRate) continue;
      this.phase -= sampleRate;
      const sample = Math.max(-1, Math.min(1, this.sum / this.count));
      this.buffer[this.offset++] = sample * (sample < 0 ? 32768 : 32767);
      this.sum = 0; this.count = 0;
      if (this.offset === this.buffer.length) {
        const pcm = this.buffer.slice().buffer;
        this.port.postMessage(pcm, [pcm]);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-input', PCMInput);
