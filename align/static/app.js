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
  spectrogramCanvas: null,
  spectrogramMeta: null,
  duration: 1,
  zoom: 1,
  viewStart: 0,
  pointerMode: '',
  pointerId: null,
  pointerStartX: 0,
  pointerStartTime: 0,
  pointerMoved: false,
  pendingHit: null,
  dragX: 0,
  dragStartView: 0,
  selection: null,
  playRangeEnd: null,
  animationFrame: null,
};

const elements = {};
const canvases = {};

window.addEventListener('DOMContentLoaded', () => {
  for (const id of [
    'status',
    'variety',
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
    'timeline-scrollbar',
    'timeline-scrollbar-spacer',
  ]) {
    elements[id] = document.getElementById(id);
  }
  for (const id of [
    'ruler',
    'waveform',
    'spectrogram',
    'feature-track',
    'projected-voicing-track',
    'candidate-overlay-track',
    'word-track',
    'phoneme-track',
    'phone-track',
  ]) {
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
  elements.audio.addEventListener('timeupdate', enforcePlaybackRange);
  elements.audio.addEventListener('ended', () => {
    state.playRangeEnd = null;
  });
  elements.timeline.addEventListener('wheel', onTimelineWheel, { passive: false });
  elements.timeline.addEventListener('pointerdown', onTimelinePointerDown);
  elements['timeline-scrollbar'].addEventListener('scroll', onTimelineScrollbarScroll);
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
    variety: elements.variety.value || 'en-US',
  }, (payload) => {
    renderPhonemicization(payload);
    clearAlignment();
    setStatus('Phonemicized');
  });
}

async function synthesize() {
  await runJsonAction('/api/synthesize', {
    text: elements.text.value,
    variety: elements.variety.value || 'en-US',
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
    variety: elements.variety.value || 'en-US',
    audio_url: state.currentAudioUrl,
  }, (payload) => {
    renderPhonemicization(payload.phonemicization);
    renderAlignment(payload);
    setStatus('Aligned with acoustic Viterbi');
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
  elements['phoneme-detail'].textContent = `${payload.variety}, ${countItems(payload.phonemes)} phonemes`;
  drawAll();
}

function renderAlignment(payload) {
  state.currentAlignment = payload;
  state.selection = null;
  state.playRangeEnd = null;
  elements.asr.textContent = [
    payload.asr_transcript ? `transcript: ${payload.asr_transcript}` : 'transcript: none',
    ...(payload.asr_segments || []).map((segment) => {
      return `${formatMs(segment.start_ms)}-${formatMs(segment.end_ms)} ${segment.text}`;
    }),
  ].join('\n');
  elements['alignment-detail'].textContent = `${payload.words.length} words, ${payload.phonemes.length} phonemes, ${payload.phones.length} phones, ${(payload.feature_tracks || []).length} features, ${(payload.projected_voicing || []).length} projected, ${(payload.candidate_overlays || []).length} candidates`;
  drawAll();
}

function clearAlignment() {
  state.currentAlignment = null;
  state.selection = null;
  state.playRangeEnd = null;
  elements.asr.textContent = '';
  elements['alignment-detail'].textContent = 'No alignment';
  drawAll();
}

async function setAudio(url, detail) {
  state.currentAudioUrl = url;
  state.currentAlignment = null;
  state.selection = null;
  state.playRangeEnd = null;
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
    state.spectrogramCanvas = null;
    state.spectrogramMeta = null;
    state.duration = state.audioBuffer.duration || 1;
    fitTimeline();
  } catch (error) {
    state.audioBuffer = null;
    state.spectrogramCanvas = null;
    state.spectrogramMeta = null;
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

function onTimelineScrollbarScroll() {
  const scrollbar = elements['timeline-scrollbar'];
  const maxScrollLeft = Math.max(0, scrollbar.scrollWidth - scrollbar.clientWidth);
  const maxStart = maxViewStart();
  const nextViewStart = maxScrollLeft > 0 ? (scrollbar.scrollLeft / maxScrollLeft) * maxStart : 0;
  const clampedViewStart = clamp(nextViewStart, 0, maxStart);
  if (Math.abs(state.viewStart - clampedViewStart) < 0.0005) return;
  state.viewStart = clampedViewStart;
  drawAll({ syncScrollbarPosition: false });
}

function onTimelinePointerDown(event) {
  event.preventDefault();
  elements.timeline.setPointerCapture?.(event.pointerId);
  state.pointerId = event.pointerId;
  state.pointerStartX = timelineX(event);
  state.pointerStartTime = clamp(xToTime(state.pointerStartX), 0, state.duration);
  state.pointerMoved = false;
  state.pendingHit = hitTestTimelineSegment(event);
  state.dragX = event.clientX;
  state.dragStartView = state.viewStart;

  if (event.button === 1 || event.altKey || event.shiftKey) {
    state.pointerMode = 'pan';
  } else {
    state.pointerMode = 'select';
  }
}

function onTimelinePointerMove(event) {
  if (!state.pointerMode || event.pointerId !== state.pointerId) return;
  const deltaX = event.clientX - state.dragX;
  if (Math.abs(timelineX(event) - state.pointerStartX) > 3) {
    state.pointerMoved = true;
  }

  if (state.pointerMode === 'pan') {
    const secondsPerPixel = viewDuration() / Math.max(1, elements.timeline.clientWidth);
    state.viewStart = clamp(state.dragStartView - deltaX * secondsPerPixel, 0, maxViewStart());
    drawAll();
    return;
  }

  if (state.pointerMode === 'select' && state.pointerMoved) {
    const currentTime = clamp(xToTime(timelineX(event)), 0, state.duration);
    selectRange(state.pointerStartTime, currentTime, { announce: false, play: false });
  }
}

function onTimelinePointerUp(event) {
  if (!state.pointerMode || event.pointerId !== state.pointerId) return;

  if (state.pointerMode === 'select') {
    if (state.pointerMoved) {
      const currentTime = clamp(xToTime(timelineX(event)), 0, state.duration);
      selectRange(state.pointerStartTime, currentTime, { announce: true, play: true });
    } else if (state.pendingHit) {
      selectSegment(state.pendingHit.kind, state.pendingHit.segment, { play: true });
    }
  }

  state.pointerMode = '';
  state.pointerId = null;
  state.pendingHit = null;
}

function startPlaybackLoop() {
  cancelAnimationFrame(state.animationFrame);
  const tick = () => {
    enforcePlaybackRange();
    drawAll();
    if (!elements.audio.paused && !elements.audio.ended) {
      state.animationFrame = requestAnimationFrame(tick);
    }
  };
  tick();
}

function drawAll(options = {}) {
  syncTimelineScrollbar(options);
  setupCanvases();
  drawRuler();
  drawWaveform();
  drawSpectrogram();
  drawTrack(canvases['feature-track'], state.currentAlignment?.feature_tracks || [], {
    kind: 'feature',
    color: featureTrackColor,
    text: '#f3f6f1',
    empty: 'Features',
  });
  drawTrack(canvases['projected-voicing-track'], state.currentAlignment?.projected_voicing || [], {
    kind: 'projected_voicing',
    color: projectedVoicingColor,
    text: '#f3f6f1',
    empty: 'Projected',
  });
  drawTrack(canvases['candidate-overlay-track'], state.currentAlignment?.candidate_overlays || [], {
    kind: 'candidate_overlay',
    color: candidateOverlayColor,
    alpha: candidateOverlayAlpha,
    text: '#f3f6f1',
    empty: 'Candidates',
  });
  drawTrack(canvases['word-track'], state.currentAlignment?.words || [], {
    kind: 'word',
    color: '#6fd2a4',
    text: '#071b12',
    empty: 'Words',
  });
  drawTrack(canvases['phoneme-track'], state.currentAlignment?.phonemes || [], {
    kind: 'phoneme',
    color: '#79b8ff',
    text: '#061728',
    empty: 'Phonemes',
    blendedChips: true,
  });
  drawTrack(canvases['phone-track'], state.currentAlignment?.phones || [], {
    kind: 'phone',
    color: '#e8c36f',
    text: '#231804',
    empty: 'Phones',
    blendedChips: true,
  });
  drawSelection();
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

function syncTimelineScrollbar({ syncScrollbarPosition = true } = {}) {
  const scrollbar = elements['timeline-scrollbar'];
  const spacer = elements['timeline-scrollbar-spacer'];
  if (!scrollbar || !spacer) return;

  const viewportWidth = Math.max(1, elements.timeline.clientWidth);
  const contentWidth = Math.max(viewportWidth, Math.round(viewportWidth * state.zoom));
  spacer.style.width = `${contentWidth}px`;

  if (!syncScrollbarPosition) return;

  const maxScrollLeft = Math.max(0, scrollbar.scrollWidth - scrollbar.clientWidth);
  const maxStart = maxViewStart();
  const nextScrollLeft = maxStart > 0 ? (state.viewStart / maxStart) * maxScrollLeft : 0;
  if (Math.abs(scrollbar.scrollLeft - nextScrollLeft) > 0.5) {
    scrollbar.scrollLeft = nextScrollLeft;
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

function drawSpectrogram() {
  const canvas = canvases.spectrogram;
  const ctx = clearCanvas(canvas);
  const width = Math.max(1, canvas.clientWidth);
  const height = canvas.clientHeight;

  if (!state.audioBuffer) {
    ctx.fillStyle = '#66727a';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('Load a WAV to draw the spectrogram', 14, Math.floor(height / 2));
    return;
  }

  ensureSpectrogram();
  if (!state.spectrogramCanvas || !state.spectrogramMeta) {
    ctx.fillStyle = '#66727a';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText('Spectrogram unavailable', 14, Math.floor(height / 2));
    return;
  }

  const sourceWidth = state.spectrogramCanvas.width;
  const sourceHeight = state.spectrogramCanvas.height;
  const startX = Math.floor((state.viewStart / state.duration) * sourceWidth);
  const viewWidth = Math.max(1, Math.ceil((viewDuration() / state.duration) * sourceWidth));
  ctx.drawImage(
    state.spectrogramCanvas,
    clamp(startX, 0, sourceWidth - 1),
    0,
    Math.min(viewWidth, sourceWidth - startX),
    sourceHeight,
    0,
    0,
    width,
    height,
  );

  drawGrid(ctx, width, height);
  ctx.fillStyle = 'rgba(12, 16, 18, 0.72)';
  ctx.fillRect(7, 7, 86, 20);
  ctx.fillStyle = '#aab5af';
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(`0-${Math.round(state.spectrogramMeta.maxHz / 1000)} kHz`, 13, 21);
}

function ensureSpectrogram() {
  if (state.spectrogramCanvas || !state.audioBuffer) return;
  const samples = state.audioBuffer.getChannelData(0);
  const sampleRate = state.audioBuffer.sampleRate;
  const fftSize = 512;
  const hopSize = 128;
  const maxHz = Math.min(8000, sampleRate / 2);
  const maxBin = Math.max(1, Math.floor((maxHz / sampleRate) * fftSize));
  const frameCount = Math.max(1, Math.floor(Math.max(0, samples.length - fftSize) / hopSize) + 1);
  const height = 192;
  const offscreen = document.createElement('canvas');
  offscreen.width = frameCount;
  offscreen.height = height;
  const ctx = offscreen.getContext('2d');
  const image = ctx.createImageData(frameCount, height);
  const windowValues = hannWindow(fftSize);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * hopSize;
    for (let i = 0; i < fftSize; i += 1) {
      re[i] = (samples[offset + i] || 0) * windowValues[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let y = 0; y < height; y += 1) {
      const normalizedY = 1 - y / Math.max(1, height - 1);
      const bin = Math.min(maxBin, Math.max(1, Math.round(normalizedY * maxBin)));
      const magnitude = Math.sqrt(re[bin] * re[bin] + im[bin] * im[bin]) / fftSize;
      const db = 20 * Math.log10(magnitude + 1e-7);
      const intensity = clamp((db + 92) / 72, 0, 1);
      const [r, g, b] = spectrogramColor(intensity);
      const index = (y * frameCount + frame) * 4;
      image.data[index] = r;
      image.data[index + 1] = g;
      image.data[index + 2] = b;
      image.data[index + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
  state.spectrogramCanvas = offscreen;
  state.spectrogramMeta = { maxHz, fftSize, hopSize };
}

function hannWindow(size) {
  const values = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    values[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / (size - 1)));
  }
  return values;
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wLenRe = Math.cos(angle);
    const wLenIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nextWRe = wRe * wLenRe - wIm * wLenIm;
        wIm = wRe * wLenIm + wIm * wLenRe;
        wRe = nextWRe;
      }
    }
  }
}

function spectrogramColor(value) {
  const stops = [
    [8, 11, 14],
    [24, 42, 64],
    [47, 96, 121],
    [111, 210, 164],
    [232, 195, 111],
    [245, 244, 205],
  ];
  const scaled = clamp(value, 0, 1) * (stops.length - 1);
  const index = Math.floor(scaled);
  const frac = scaled - index;
  const left = stops[index];
  const right = stops[Math.min(stops.length - 1, index + 1)];
  return [
    Math.round(left[0] + (right[0] - left[0]) * frac),
    Math.round(left[1] + (right[1] - left[1]) * frac),
    Math.round(left[2] + (right[2] - left[2]) * frac),
  ];
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
    const selected = isSelectedSegment(options.kind, segment);
    const x = timeToX(segment.start_ms / 1000);
    const right = timeToX(segment.end_ms / 1000);
    const w = Math.max(1, right - x);
    if (right < 0 || x > width) continue;
    const fillColor = typeof options.color === 'function' ? options.color(segment) : options.color;
    const alpha = typeof options.alpha === 'function' ? options.alpha(segment) : 0.92;
    const y = 8;
    const h = height - 16;
    if (options.blendedChips) {
      const bleed = Math.min(10, Math.max(2, w * 0.16));
      const gradient = ctx.createLinearGradient(x - bleed, 0, right + bleed, 0);
      const chipColor = selected ? '#f3f6f1' : fillColor;
      gradient.addColorStop(0, rgba(chipColor, 0));
      gradient.addColorStop(0.18, rgba(chipColor, alpha * 0.76));
      gradient.addColorStop(0.5, rgba(chipColor, alpha));
      gradient.addColorStop(0.82, rgba(chipColor, alpha * 0.76));
      gradient.addColorStop(1, rgba(chipColor, 0));
      ctx.fillStyle = gradient;
      ctx.fillRect(x - bleed, y, w + bleed * 2, h);
    } else {
      ctx.fillStyle = selected ? '#f3f6f1' : fillColor;
      ctx.globalAlpha = alpha;
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1;
    }
    ctx.globalAlpha = 1;
    if (options.blendedChips) {
      if (selected) {
        ctx.strokeStyle = '#ef8c86';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2));
        ctx.lineWidth = 1;
      }
    } else {
      ctx.strokeStyle = selected ? '#ef8c86' : '#0b0e10';
      ctx.lineWidth = selected ? 2 : 1;
      ctx.strokeRect(x, y, w, h);
      ctx.lineWidth = 1;
    }
    if (w > 16) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y, Math.max(0, w - 4), h);
      ctx.clip();
      ctx.fillStyle = selected ? '#0b0e10' : options.text;
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
      if (options.blendedChips) {
        ctx.textAlign = 'center';
        ctx.fillText(segment.label || segment.text || '', x + w / 2, Math.floor(height / 2) + 4);
        ctx.textAlign = 'start';
      } else {
        ctx.fillText(segment.label || segment.text || '', x + 5, Math.floor(height / 2) + 4);
      }
      ctx.restore();
    }
  }
}

function rgba(color, alpha) {
  if (!color.startsWith('#')) return color;
  const hex = color.slice(1);
  const values =
    hex.length === 3
      ? hex.split('').map((digit) => Number.parseInt(`${digit}${digit}`, 16))
      : [hex.slice(0, 2), hex.slice(2, 4), hex.slice(4, 6)].map((pair) => Number.parseInt(pair, 16));
  return `rgba(${values[0]}, ${values[1]}, ${values[2]}, ${clamp(alpha, 0, 1)})`;
}

function featureTrackColor(segment) {
  if (segment.kind === 'silence') return '#2c3337';
  if (segment.kind === 'voiced') return '#5bc6ff';
  if (segment.kind === 'unvoiced') return '#c3a4ff';
  return '#66727a';
}

function projectedVoicingColor(segment) {
  if (segment.kind === 'voiced') return '#2f9cc9';
  if (segment.kind === 'unvoiced') return '#9f7fd2';
  return '#66727a';
}

function candidateOverlayColor(segment) {
  if (segment.source === 'reverse_snipper') return '#e8c36f';
  if (segment.source === 'syllable_nucleus') return '#ef8c86';
  if (segment.source === 'frication') return '#c3a4ff';
  if (segment.source === 'vowel_trajectory') return '#6fd2a4';
  return '#66727a';
}

function candidateOverlayAlpha(segment) {
  const confidence = Number(segment.confidence);
  if (!Number.isFinite(confidence)) return 0.72;
  return clamp(0.34 + confidence * 0.58, 0.34, 0.92);
}

function drawSelection() {
  const selection = state.selection;
  if (!selection) return;

  const startX = timeToX(selection.start);
  const endX = timeToX(selection.end);
  for (const canvas of Object.values(canvases)) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (endX < 0 || startX > width) continue;

    const x1 = clamp(startX, 0, width);
    const x2 = clamp(endX, 0, width);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0, 0, 0, 0.24)';
    if (x1 > 0) ctx.fillRect(0, 0, x1, height);
    if (x2 < width) ctx.fillRect(x2, 0, width - x2, height);
    ctx.fillStyle = 'rgba(243, 246, 241, 0.10)';
    ctx.fillRect(x1, 0, Math.max(1, x2 - x1), height);
    ctx.strokeStyle = '#f3f6f1';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1 + 0.5, 0);
    ctx.lineTo(x1 + 0.5, height);
    ctx.moveTo(x2 - 0.5, 0);
    ctx.lineTo(x2 - 0.5, height);
    ctx.stroke();
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

function selectSegment(kind, segment, { play = false } = {}) {
  const start = clamp(segment.start_ms / 1000, 0, state.duration);
  const end = clamp(Math.max(segment.end_ms / 1000, start + 0.001), 0, state.duration);
  state.selection = {
    kind,
    key: segmentKey(kind, segment),
    start,
    end,
    label: segment.label || segment.text || kind,
  };
  setStatus(`Selected ${kind} ${state.selection.label} ${formatRange(start, end)}`);
  drawAll();
  if (play) playSelection();
}

function selectRange(start, end, { announce = true, play = false } = {}) {
  const left = clamp(Math.min(start, end), 0, state.duration);
  const right = clamp(Math.max(start, end), 0, state.duration);
  const minDuration = Math.min(0.015, state.duration);
  const normalizedEnd = Math.min(state.duration, Math.max(right, left + minDuration));
  const normalizedStart = normalizedEnd > left ? left : Math.max(0, normalizedEnd - minDuration);
  state.selection = {
    kind: 'range',
    key: '',
    start: normalizedStart,
    end: normalizedEnd,
    label: 'range',
  };
  if (announce) {
    setStatus(`Selected range ${formatRange(state.selection.start, state.selection.end)}`);
  }
  drawAll();
  if (play) playSelection();
}

async function playSelection() {
  if (!state.selection || !elements.audio.src) return;
  const start = state.selection.start;
  const end = Math.max(state.selection.end, start + 0.001);
  state.playRangeEnd = end;
  elements.audio.currentTime = start;
  try {
    await elements.audio.play();
  } catch (error) {
    state.playRangeEnd = null;
    setStatus(error.message || String(error), 'error');
  }
}

function enforcePlaybackRange() {
  if (state.playRangeEnd == null) return;
  if (elements.audio.currentTime < state.playRangeEnd - 0.004) return;
  elements.audio.pause();
  elements.audio.currentTime = state.playRangeEnd;
  state.playRangeEnd = null;
  drawAll();
}

function hitTestTimelineSegment(event) {
  const kind = trackKindForTarget(event.target);
  if (!kind) return null;
  const alignment = state.currentAlignment;
  if (!alignment) return null;
  const timeMs = xToTime(timelineX(event)) * 1000;
  const segments = alignmentSegments(alignment, kind);
  const segment = segments.find((candidate) => {
    return timeMs >= candidate.start_ms && timeMs <= candidate.end_ms;
  });
  return segment ? { kind, segment } : null;
}

function alignmentSegments(alignment, kind) {
  if (kind === 'feature') return alignment.feature_tracks || [];
  if (kind === 'projected_voicing') return alignment.projected_voicing || [];
  if (kind === 'candidate_overlay') return alignment.candidate_overlays || [];
  return alignment[`${kind}s`] || [];
}

function trackKindForTarget(target) {
  if (!target || !target.id) return '';
  if (target.id === 'feature-track') return 'feature';
  if (target.id === 'projected-voicing-track') return 'projected_voicing';
  if (target.id === 'candidate-overlay-track') return 'candidate_overlay';
  if (target.id === 'word-track') return 'word';
  if (target.id === 'phoneme-track') return 'phoneme';
  if (target.id === 'phone-track') return 'phone';
  return '';
}

function isSelectedSegment(kind, segment) {
  if (!state.selection || state.selection.kind !== kind) return false;
  return state.selection.key === segmentKey(kind, segment);
}

function segmentKey(kind, segment) {
  if (kind === 'feature' || kind === 'projected_voicing' || kind === 'candidate_overlay') {
    return String(segment.index);
  }
  if (kind === 'word') return String(segment.index);
  return `${segment.word_index}:${segment.index}:${segment.token_id || segment.label || ''}`;
}

function timelineX(event) {
  const rect = elements.timeline.getBoundingClientRect();
  return clamp(event.clientX - rect.left, 0, rect.width);
}

function formatRange(start, end) {
  return `${formatSeconds(start)}-${formatSeconds(end)}`;
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
