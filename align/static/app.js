const state = {
  audioContext: null,
  source: null,
  processor: null,
  mediaStream: null,
  recordingChunks: [],
  recordingSampleRate: 0,
  recording: false,
  irVisible: false,
  currentAudioUrl: '',
  currentPhonemicization: null,
  currentAlignment: null,
  styletts2VoiceDir: 'voices/styletts2',
  audioBuffer: null,
  duration: 1,
  zoom: 1,
  viewStart: 0,
  dragging: false,
  dragX: 0,
  dragStartView: 0,
  animationFrame: null,
};

const elements = {};
const canvases = {};

window.addEventListener('DOMContentLoaded', () => {
  for (const id of [
    'status',
    'variant',
    'text',
    'backend',
    'styletts2-voice',
    'refresh-voices',
    'voice-detail',
    'phonemicize',
    'synthesize',
    'align',
    'wav-file',
    'record',
    'stop-recording',
    'audio',
    'audio-detail',
    'phoneme-detail',
    'alignment-detail',
    'phonemes',
    'phones',
    'syllables',
    'warnings',
    'asr',
    'ir',
    'toggle-ir',
    'zoom-in',
    'zoom-out',
    'fit',
    'timeline',
  ]) {
    elements[id] = document.getElementById(id);
  }
  for (const id of ['ruler', 'waveform', 'word-track', 'phoneme-track', 'phone-track']) {
    canvases[id] = document.getElementById(id);
  }

  elements.phonemicize.addEventListener('click', phonemicize);
  elements.synthesize.addEventListener('click', synthesize);
  elements.align.addEventListener('click', alignAudio);
  elements.backend.addEventListener('change', syncVoiceSelector);
  elements['refresh-voices'].addEventListener('click', loadStyleTts2Voices);
  elements['wav-file'].addEventListener('change', uploadSelectedFile);
  elements.record.addEventListener('click', startRecording);
  elements['stop-recording'].addEventListener('click', stopRecording);
  elements['toggle-ir'].addEventListener('click', toggleIr);
  elements['zoom-in'].addEventListener('click', () => zoomAtCenter(1.45));
  elements['zoom-out'].addEventListener('click', () => zoomAtCenter(1 / 1.45));
  elements.fit.addEventListener('click', fitTimeline);
  elements.audio.addEventListener('loadedmetadata', updateAudioDuration);
  elements.audio.addEventListener('play', startPlaybackLoop);
  elements.audio.addEventListener('pause', drawAll);
  elements.audio.addEventListener('seeked', drawAll);
  elements.timeline.addEventListener('wheel', onTimelineWheel, { passive: false });
  elements.timeline.addEventListener('pointerdown', onTimelinePointerDown);
  window.addEventListener('pointermove', onTimelinePointerMove);
  window.addEventListener('pointerup', onTimelinePointerUp);
  window.addEventListener('resize', drawAll);

  fitTimeline();
  loadStyleTts2Voices();
  syncVoiceSelector();
  drawAll();
});

async function phonemicize() {
  await runJsonAction('/api/phonemicize', {
    text: elements.text.value,
    variant: elements.variant.value || 'en-US',
  }, (payload) => {
    renderPhonemicization(payload);
    clearAlignment();
    setStatus('Phonemicized');
  });
}

async function synthesize() {
  await runJsonAction('/api/synthesize', {
    text: elements.text.value,
    variant: elements.variant.value || 'en-US',
    backend: elements.backend.value,
    styletts2_voice: elements['styletts2-voice'].value || null,
  }, async (payload) => {
    renderPhonemicization(payload.phonemicization);
    await setAudio(payload.audio_url, `${payload.duration_ms} ms, ${payload.samples} samples`);
    setStatus(`Synthesized with ${elements.backend.value}`);
    await alignAudio();
  });
}

async function loadStyleTts2Voices() {
  try {
    const selected = elements['styletts2-voice'].value;
    const response = await fetch('/api/styletts2/voices');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    state.styletts2VoiceDir = payload.directory || 'voices/styletts2';
    elements['styletts2-voice'].replaceChildren(
      optionElement('', 'default reference'),
      ...(payload.voices || []).map((voice) => optionElement(voice.id, voice.label)),
    );
    if ([...elements['styletts2-voice'].options].some((option) => option.value === selected)) {
      elements['styletts2-voice'].value = selected;
    }
    elements['voice-detail'].textContent = (payload.voices || []).length
      ? `${payload.voices.length} WAV voice${payload.voices.length === 1 ? '' : 's'} in ${state.styletts2VoiceDir}`
      : `Drop WAVs in ${state.styletts2VoiceDir}`;
  } catch (error) {
    elements['voice-detail'].textContent = error.message || String(error);
  } finally {
    syncVoiceSelector();
  }
}

function optionElement(value, label) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  return option;
}

function syncVoiceSelector() {
  const enabled = elements.backend.value === 'styletts2';
  elements['styletts2-voice'].disabled = !enabled;
  elements['refresh-voices'].disabled = false;
}

async function alignAudio() {
  if (!state.currentAudioUrl) {
    setStatus('Load or synthesize a WAV first', 'error');
    return;
  }
  await runJsonAction('/api/align', {
    text: elements.text.value,
    variant: elements.variant.value || 'en-US',
    audio_url: state.currentAudioUrl,
  }, (payload) => {
    renderPhonemicization(payload.phonemicization);
    renderAlignment(payload);
    setStatus('Aligned with ASR timings');
  });
}

async function runJsonAction(url, body, onSuccess) {
  setBusy(true);
  setStatus('Working');
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    await onSuccess(payload);
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  } finally {
    setBusy(false);
  }
}

async function uploadSelectedFile(event) {
  const [file] = event.target.files || [];
  if (!file) return;
  await uploadWav(file, file.name);
  event.target.value = '';
}

async function uploadWav(blob, filename) {
  setBusy(true);
  setStatus('Uploading');
  try {
    const form = new FormData();
    form.append('file', blob, filename || 'recording.wav');
    const response = await fetch('/api/audio/upload', {
      method: 'POST',
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    await setAudio(payload.audio_url, `${payload.bytes} bytes`);
    setStatus('Audio ready');
    await alignAudio();
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  } finally {
    setBusy(false);
  }
}

async function startRecording() {
  if (state.recording) return;
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    state.audioContext = new AudioContext();
    state.recordingSampleRate = state.audioContext.sampleRate;
    state.source = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.recordingChunks = [];
    state.processor.onaudioprocess = (event) => {
      if (!state.recording) return;
      const input = event.inputBuffer.getChannelData(0);
      state.recordingChunks.push(new Float32Array(input));
    };
    state.source.connect(state.processor);
    state.processor.connect(state.audioContext.destination);
    state.recording = true;
    elements.record.disabled = true;
    elements['stop-recording'].disabled = false;
    setStatus('Recording');
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  }
}

async function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  elements.record.disabled = false;
  elements['stop-recording'].disabled = true;

  if (state.processor) {
    state.processor.disconnect();
    state.processor.onaudioprocess = null;
    state.processor = null;
  }
  if (state.source) {
    state.source.disconnect();
    state.source = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  if (state.audioContext) {
    await state.audioContext.close();
    state.audioContext = null;
  }

  const wav = encodeWav(flattenChunks(state.recordingChunks), state.recordingSampleRate || 48000);
  state.recordingChunks = [];
  await uploadWav(wav, `recording-${Date.now()}.wav`);
}

function flattenChunks(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const samples = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  return samples;
}

function encodeWav(samples, sampleRate) {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeAscii(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function renderPhonemicization(payload) {
  state.currentPhonemicization = payload;
  elements.phonemes.textContent = payload.phonemes || '';
  elements.phones.textContent = payload.phones || '';
  elements.syllables.textContent = (payload.syllables || [])
    .map((syllable) => `${syllable.label} (${syllable.stress})`)
    .join(' / ');
  elements.warnings.textContent = (payload.warnings || [])
    .map((warning) => `${warning.token}: ${warning.message}`)
    .join('\n');
  elements.ir.textContent = JSON.stringify(payload.ir, null, 2);
  elements['phoneme-detail'].textContent = `${payload.variant}, ${countItems(payload.phonemes)} phonemes`;
  drawAll();
}

function renderAlignment(payload) {
  state.currentAlignment = payload;
  elements.asr.textContent = [
    payload.asr_transcript ? `transcript: ${payload.asr_transcript}` : 'transcript: none',
    ...(payload.asr_segments || []).map((segment) => {
      return `${formatMs(segment.start_ms)}-${formatMs(segment.end_ms)} ${segment.text}`;
    }),
  ].join('\n');
  elements['alignment-detail'].textContent = `${payload.words.length} words, ${payload.phonemes.length} phonemes, ${payload.phones.length} phones`;
  drawAll();
}

function clearAlignment() {
  state.currentAlignment = null;
  elements.asr.textContent = '';
  elements['alignment-detail'].textContent = 'No alignment';
  drawAll();
}

async function setAudio(url, detail) {
  state.currentAudioUrl = url;
  state.currentAlignment = null;
  elements.audio.src = url;
  elements.audio.load();
  elements['audio-detail'].textContent = detail || 'Audio ready';
  await loadWaveform(url);
}

async function loadWaveform(url) {
  try {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const context = new AudioContext();
    state.audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
    await context.close();
    state.duration = state.audioBuffer.duration || 1;
    fitTimeline();
  } catch (error) {
    state.audioBuffer = null;
    state.duration = 1;
    setStatus(`Waveform decode failed: ${error.message || error}`, 'error');
    drawAll();
  }
}

function updateAudioDuration() {
  if (!Number.isFinite(elements.audio.duration)) return;
  elements['audio-detail'].textContent = `${elements.audio.duration.toFixed(2)} s`;
}

function fitTimeline() {
  state.zoom = 1;
  state.viewStart = 0;
  drawAll();
}

function zoomAtCenter(multiplier) {
  const rect = elements.timeline.getBoundingClientRect();
  zoomAt(rect.width / 2, multiplier);
}

function zoomAt(x, multiplier) {
  const oldDuration = viewDuration();
  const centerTime = xToTime(x);
  state.zoom = clamp(state.zoom * multiplier, 1, 240);
  const newDuration = viewDuration();
  const fraction = x / Math.max(1, elements.timeline.clientWidth);
  state.viewStart = clamp(centerTime - newDuration * fraction, 0, maxViewStart());
  if (Math.abs(oldDuration - newDuration) > 0.0001) drawAll();
}

function onTimelineWheel(event) {
  event.preventDefault();
  const rect = elements.timeline.getBoundingClientRect();
  const multiplier = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  zoomAt(event.clientX - rect.left, multiplier);
}

function onTimelinePointerDown(event) {
  elements.timeline.setPointerCapture?.(event.pointerId);
  state.dragging = true;
  state.dragX = event.clientX;
  state.dragStartView = state.viewStart;
}

function onTimelinePointerMove(event) {
  if (!state.dragging) return;
  const deltaX = event.clientX - state.dragX;
  const secondsPerPixel = viewDuration() / Math.max(1, elements.timeline.clientWidth);
  state.viewStart = clamp(state.dragStartView - deltaX * secondsPerPixel, 0, maxViewStart());
  drawAll();
}

function onTimelinePointerUp() {
  state.dragging = false;
}

function startPlaybackLoop() {
  cancelAnimationFrame(state.animationFrame);
  const tick = () => {
    drawAll();
    if (!elements.audio.paused && !elements.audio.ended) {
      state.animationFrame = requestAnimationFrame(tick);
    }
  };
  tick();
}

function drawAll() {
  setupCanvases();
  drawRuler();
  drawWaveform();
  drawTrack(canvases['word-track'], state.currentAlignment?.words || [], {
    color: '#6fd2a4',
    text: '#071b12',
    empty: 'Words',
  });
  drawTrack(canvases['phoneme-track'], state.currentAlignment?.phonemes || [], {
    color: '#79b8ff',
    text: '#061728',
    empty: 'Phonemes',
  });
  drawTrack(canvases['phone-track'], state.currentAlignment?.phones || [], {
    color: '#e8c36f',
    text: '#231804',
    empty: 'Phones',
  });
  drawPlayhead();
}

function setupCanvases() {
  const pixelRatio = window.devicePixelRatio || 1;
  for (const canvas of Object.values(canvases)) {
    const cssWidth = canvas.clientWidth || canvas.parentElement.clientWidth || 300;
    const cssHeight = Number(canvas.getAttribute('height')) || canvas.clientHeight || 40;
    const width = Math.max(1, Math.floor(cssWidth * pixelRatio));
    const height = Math.max(1, Math.floor(cssHeight * pixelRatio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = '#0c1012';
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawGrid(ctx, canvas.clientWidth, canvas.clientHeight);
  return ctx;
}

function drawGrid(ctx, width, height) {
  const seconds = tickSeconds();
  const firstTick = Math.floor(state.viewStart / seconds) * seconds;
  ctx.strokeStyle = '#293238';
  ctx.lineWidth = 1;
  for (let time = firstTick; time <= state.viewStart + viewDuration(); time += seconds) {
    const x = timeToX(time);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function drawRuler() {
  const canvas = canvases.ruler;
  const ctx = clearCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.fillStyle = '#11171a';
  ctx.fillRect(0, 0, width, height);
  const seconds = tickSeconds();
  const firstTick = Math.floor(state.viewStart / seconds) * seconds;
  ctx.strokeStyle = '#46525a';
  ctx.fillStyle = '#aab5af';
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  for (let time = firstTick; time <= state.viewStart + viewDuration(); time += seconds) {
    const x = timeToX(time);
    ctx.beginPath();
    ctx.moveTo(x, height - 10);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.fillText(formatSeconds(time), x + 4, 14);
  }
}

function drawWaveform() {
  const canvas = canvases.waveform;
  const ctx = clearCanvas(canvas);
  const width = Math.max(1, canvas.clientWidth);
  const height = canvas.clientHeight;
  const middle = height / 2;
  ctx.strokeStyle = '#46525a';
  ctx.beginPath();
  ctx.moveTo(0, middle);
  ctx.lineTo(width, middle);
  ctx.stroke();

  if (!state.audioBuffer) {
    ctx.fillStyle = '#66727a';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('Load a WAV to draw the waveform', 14, middle - 8);
    return;
  }

  const samples = state.audioBuffer.getChannelData(0);
  const sampleRate = state.audioBuffer.sampleRate;
  const startSample = Math.max(0, Math.floor(state.viewStart * sampleRate));
  const endSample = Math.min(samples.length, Math.ceil((state.viewStart + viewDuration()) * sampleRate));
  const samplesPerPixel = Math.max(1, Math.floor((endSample - startSample) / width));

  ctx.strokeStyle = '#6fd2a4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const start = startSample + x * samplesPerPixel;
    const end = Math.min(endSample, start + samplesPerPixel);
    let min = 0;
    let max = 0;
    for (let index = start; index < end; index += 1) {
      const sample = samples[index] || 0;
      if (sample < min) min = sample;
      if (sample > max) max = sample;
    }
    ctx.moveTo(x + 0.5, middle + min * middle * 0.92);
    ctx.lineTo(x + 0.5, middle + max * middle * 0.92);
  }
  ctx.stroke();
}

function drawTrack(canvas, segments, options) {
  const ctx = clearCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!segments.length) {
    ctx.fillStyle = '#66727a';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText(options.empty, 14, Math.floor(height / 2) + 4);
    return;
  }

  for (const segment of segments) {
    const x = timeToX(segment.start_ms / 1000);
    const right = timeToX(segment.end_ms / 1000);
    const w = Math.max(1, right - x);
    if (right < 0 || x > width) continue;
    ctx.fillStyle = options.color;
    ctx.globalAlpha = 0.92;
    ctx.fillRect(x, 8, w, height - 16);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#0b0e10';
    ctx.strokeRect(x, 8, w, height - 16);
    if (w > 16) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, 8, Math.max(0, w - 4), height - 16);
      ctx.clip();
      ctx.fillStyle = options.text;
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(segment.label || segment.text || '', x + 5, Math.floor(height / 2) + 4);
      ctx.restore();
    }
  }
}

function drawPlayhead() {
  const time = elements.audio.currentTime || 0;
  if (time < state.viewStart || time > state.viewStart + viewDuration()) return;
  const x = timeToX(time);
  for (const canvas of Object.values(canvases)) {
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = '#ef8c86';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.clientHeight);
    ctx.stroke();
  }
}

function toggleIr() {
  state.irVisible = !state.irVisible;
  elements.ir.hidden = !state.irVisible;
  elements['toggle-ir'].textContent = state.irVisible ? 'Hide' : 'Show';
}

function setBusy(busy) {
  for (const id of ['phonemicize', 'synthesize', 'align']) {
    elements[id].disabled = busy;
  }
}

function setStatus(message, tone = '') {
  elements.status.textContent = message;
  if (tone) {
    elements.status.dataset.tone = tone;
  } else {
    delete elements.status.dataset.tone;
  }
}

function viewDuration() {
  return Math.max(0.05, state.duration / state.zoom);
}

function maxViewStart() {
  return Math.max(0, state.duration - viewDuration());
}

function timeToX(time) {
  return ((time - state.viewStart) / viewDuration()) * Math.max(1, elements.timeline.clientWidth);
}

function xToTime(x) {
  return state.viewStart + (x / Math.max(1, elements.timeline.clientWidth)) * viewDuration();
}

function tickSeconds() {
  const targetTicks = Math.max(4, elements.timeline.clientWidth / 120);
  const rough = viewDuration() / targetTicks;
  const bases = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60];
  return bases.find((base) => base >= rough) || 120;
}

function formatSeconds(value) {
  if (value < 10) return `${value.toFixed(2)}s`;
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function formatMs(value) {
  return `${(value / 1000).toFixed(2)}s`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function countItems(text) {
  return text ? text.split(/\s+/).filter(Boolean).length : 0;
}
