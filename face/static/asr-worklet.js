class MortarAsrCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input?.length) {
      const samples = new Float32Array(input.length);
      samples.set(input);
      this.port.postMessage(samples, [samples.buffer]);
    }
    return true;
  }
}

registerProcessor('mortar-asr-capture', MortarAsrCaptureProcessor);
