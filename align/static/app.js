const PREFERENCES_STORAGE_KEY = 'mortar-align.preferences.v1';
const DEFAULT_PREFERENCES = {
  backend: 'styletts2',
  styletts2Voice: '',
  styletts2Style: '',
  variety: 'en-US',
};

const state = {
  audioContext: null,
  source: null,
  processor: null,
  mediaStream: null,
  recordingChunks: [],
  recordingSampleRate: 0,
  recording: false,
  recordingTarget: '',
  irVisible: false,
  currentAudioUrl: '',
  currentPhonemicization: null,
  currentAlignment: null,
  alignInProgress: false,
  inputVersion: 0,
  activeActionCount: 0,
  inputsChangedDuringAction: false,
  nextActionId: 0,
  latestActionId: 0,
  styletts2VoiceDir: 'voices/styletts2',
  pendingStyletts2Voice: '',
  pendingStyletts2Style: '',
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
  audioSeeking: false,
};

const elements = {};
const canvases = {};
const ICON_FALLBACKS = {
  'chevron-down': '\u25be',
  'chevron-right': '\u25b8',
  circle: '\u25cf',
  maximize: '\u26f6',
  pause: '\u23f8',
  play: '\u25b6',
  square: '\u25a0',
  upload: '\u21e7',
  zap: '\u26a1',
  'zoom-in': '+',
  'zoom-out': '-',
  'refresh-cw': '\u21bb',
};

window.addEventListener('DOMContentLoaded', () => {
  for (const id of [
    'status',
    'variety',
    'text',
    'backend',
    'styletts2-voice',
    'styletts2-style',
    'refresh-voices',
    'styletts2-voice-file',
    'record-styletts2-voice',
    'voice-detail',
    'phonemicize',
    'synthesize',
    'align',
    'wav-file',
    'record',
    'audio',
    'audio-play-toggle',
    'audio-position',
    'audio-progress',
    'audio-progress-fill',
    'audio-progress-thumb',
    'audio-duration',
    'audio-detail',
    'phoneme-detail',
    'alignment-detail',
    'phonemes',
    'syllables',
    'asr',
    'ir',
    'toggle-ir',
    'zoom-in',
    'zoom-out',
    'fit',
    'timeline',
    'timeline-content',
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
  elements.text.addEventListener('input', markActionInputsChanged);
  elements.text.addEventListener('change', markActionInputsChanged);
  elements.synthesize.addEventListener('click', synthesize);
  elements.align.addEventListener('click', alignAudio);
  elements.backend.addEventListener('change', () => {
    syncVoiceSelector();
    savePreferences();
    markActionInputsChanged();
  });
  elements['styletts2-voice'].addEventListener('change', () => {
    state.pendingStyletts2Voice = elements['styletts2-voice'].value;
    savePreferences();
    markActionInputsChanged();
  });
  elements['styletts2-style'].addEventListener('change', () => {
    state.pendingStyletts2Style = elements['styletts2-style'].value;
    savePreferences();
    markActionInputsChanged();
  });
  elements.variety.addEventListener('change', () => {
    savePreferences();
    markActionInputsChanged();
  });
  elements.variety.addEventListener('input', () => {
    savePreferences();
    markActionInputsChanged();
  });
  elements['refresh-voices'].addEventListener('click', loadStyleTts2Voices);
  elements['styletts2-voice-file'].addEventListener('change', uploadSelectedStyleTts2VoiceFile);
  elements['wav-file'].addEventListener('change', uploadSelectedFile);
  elements.record.addEventListener('click', () => toggleRecording('audio'));
  elements['record-styletts2-voice'].addEventListener('click', () => toggleRecording('styletts2Voice'));
  elements['toggle-ir'].addEventListener('click', toggleIr);
  elements['zoom-in'].addEventListener('click', () => zoomAtCenter(1.45));
  elements['zoom-out'].addEventListener('click', () => zoomAtCenter(1 / 1.45));
  elements.fit.addEventListener('click', fitTimeline);
  elements['audio-play-toggle'].addEventListener('click', toggleAudioPlayback);
  elements['audio-progress'].addEventListener('pointerdown', onAudioProgressPointerDown);
  elements['audio-progress'].addEventListener('pointermove', onAudioProgressPointerMove);
  elements['audio-progress'].addEventListener('pointerup', onAudioProgressPointerUp);
  elements['audio-progress'].addEventListener('pointercancel', onAudioProgressPointerUp);
  elements['audio-progress'].addEventListener('keydown', onAudioProgressKeyDown);
  elements.audio.addEventListener('loadedmetadata', updateAudioDuration);
  elements.audio.addEventListener('play', () => {
    updateAudioTransport();
    startPlaybackLoop();
  });
  elements.audio.addEventListener('pause', () => {
    updateAudioTransport();
    drawAll();
  });
  elements.audio.addEventListener('seeked', () => {
    updateAudioTransport();
    drawAll();
  });
  elements.audio.addEventListener('timeupdate', () => {
    enforcePlaybackRange();
    updateAudioTransport();
  });
  elements.audio.addEventListener('ended', () => {
    state.playRangeEnd = null;
    updateAudioTransport();
    drawAll();
  });
  elements.timeline.addEventListener('wheel', onTimelineWheel, { passive: false });
  elements.timeline.addEventListener('pointerdown', onTimelinePointerDown);
  elements.timeline.addEventListener('scroll', onTimelineScroll);
  window.addEventListener('pointermove', onTimelinePointerMove);
  window.addEventListener('pointerup', onTimelinePointerUp);
  window.addEventListener('resize', drawAll);

  fitTimeline();
  applyPreferences();
  loadStyleTts2Voices();
  syncVoiceSelector();
  updateAudioTransport();
  updateRecordingControls();
  updateActionButtons();
  renderLucideIcons();
  drawAll();
});

function iconMarkup(iconName, label = '') {
  const fallback = ICON_FALLBACKS[iconName] || '';
  const labelMarkup = label ? `<span>${label}</span>` : '';
  return `<i data-lucide="${iconName}" aria-hidden="true">${fallback}</i>${labelMarkup}`;
}

function renderLucideIcons() {
  window.lucide?.createIcons?.({
    attrs: {
      'aria-hidden': 'true',
      focusable: 'false',
    },
  });
}

function setButtonIcon(button, iconName, ariaLabel, title = ariaLabel, visibleLabel = '') {
  if (
    button.dataset.icon === iconName
    && button.getAttribute('aria-label') === ariaLabel
    && button.title === title
    && (button.dataset.visibleLabel || '') === visibleLabel
  ) {
    return;
  }
  button.innerHTML = iconMarkup(iconName, visibleLabel);
  button.dataset.icon = iconName;
  button.dataset.visibleLabel = visibleLabel;
  button.title = title;
  button.setAttribute('aria-label', ariaLabel);
  renderLucideIcons();
}

function loadPreferences() {
  try {
    const raw = window.localStorage?.getItem(PREFERENCES_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_PREFERENCES };
    return { ...DEFAULT_PREFERENCES, ...JSON.parse(raw) };
  } catch (_error) {
    return { ...DEFAULT_PREFERENCES };
  }
}

function savePreferences() {
  try {
    window.localStorage?.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify({
      backend: elements.backend.value || DEFAULT_PREFERENCES.backend,
      styletts2Voice: state.pendingStyletts2Voice || elements['styletts2-voice'].value || '',
      styletts2Style: state.pendingStyletts2Style || elements['styletts2-style'].value || '',
      variety: elements.variety.value || DEFAULT_PREFERENCES.variety,
    }));
  } catch (_error) {
    // Private browsing or locked-down storage should not break the aligner.
  }
}

function applyPreferences() {
  const preferences = loadPreferences();
  if (selectHasValue(elements.backend, preferences.backend)) {
    elements.backend.value = preferences.backend;
  } else {
    elements.backend.value = DEFAULT_PREFERENCES.backend;
  }
  elements.variety.value = preferences.variety || DEFAULT_PREFERENCES.variety;
  state.pendingStyletts2Voice = preferences.styletts2Voice || '';
  state.pendingStyletts2Style = preferences.styletts2Style || '';
}

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
  const synthesisInputVersion = state.inputVersion;
  await runJsonAction('/api/synthesize', {
    text: elements.text.value,
    variety: elements.variety.value || 'en-US',
    backend: elements.backend.value,
    styletts2_voice: elements['styletts2-voice'].value || null,
    styletts2_style: elements['styletts2-style'].value || null,
  }, async (payload) => {
    renderPhonemicization(payload.phonemicization);
    await setAudio(payload.audio_url, `${payload.duration_ms} ms, ${payload.samples} samples`);
    if (synthesisInputVersion !== state.inputVersion) {
      setStatus('Inputs changed; run again');
      return;
    }
    setStatus(`Synthesized with ${elements.backend.value}`);
    await alignAudio();
  });
}

async function loadStyleTts2Voices() {
  try {
    const selectedVoice = state.pendingStyletts2Voice || elements['styletts2-voice'].value;
    const selectedStyle = state.pendingStyletts2Style || elements['styletts2-style'].value;
    const response = await fetch('/api/styletts2/voices');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    state.styletts2VoiceDir = payload.directory || 'voices/styletts2';
    elements['styletts2-voice'].replaceChildren(
      optionElement('', 'default speaker'),
      ...(payload.voices || []).map((voice) => optionElement(voice.id, voice.label)),
    );
    elements['styletts2-style'].replaceChildren(
      optionElement('', 'default style'),
      ...(payload.voices || []).map((voice) => optionElement(voice.id, voice.label)),
    );
    if (selectHasValue(elements['styletts2-voice'], selectedVoice)) {
      elements['styletts2-voice'].value = selectedVoice;
    }
    if (selectHasValue(elements['styletts2-style'], selectedStyle)) {
      elements['styletts2-style'].value = selectedStyle;
    }
    elements['voice-detail'].textContent = (payload.voices || []).length
      ? `${payload.voices.length} WAV${payload.voices.length === 1 ? '' : 's'} in ${state.styletts2VoiceDir}`
      : `Drop WAVs in ${state.styletts2VoiceDir}`;
  } catch (error) {
    elements['voice-detail'].textContent = error.message || String(error);
  } finally {
    syncVoiceSelector();
  }
}

function selectHasValue(select, value) {
  return [...select.options].some((option) => option.value === value);
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
  elements['styletts2-style'].disabled = !enabled;
  elements['refresh-voices'].disabled = false;
  elements['styletts2-voice-file'].disabled = !enabled;
  updateRecordingControls();
}

async function alignAudio() {
  if (!state.currentAudioUrl) {
    setStatus('Load or synthesize a WAV first', 'error');
    return;
  }
  const previousAlignmentDetail = elements['alignment-detail'].textContent || 'No alignment';
  setAlignProgress(true);
  elements['alignment-detail'].textContent = 'Aligning audio...';
  const aligned = await runJsonAction('/api/align', {
    text: elements.text.value,
    variety: elements.variety.value || 'en-US',
    audio_url: state.currentAudioUrl,
  }, (payload) => {
    renderPhonemicization(payload.phonemicization);
    renderAlignment(payload);
    setStatus('Aligned with acoustic Viterbi');
  }, {
    status: 'Aligning audio',
  });
  setAlignProgress(false);
  if (!aligned) {
    elements['alignment-detail'].textContent = previousAlignmentDetail;
  }
}

async function runJsonAction(url, body, onSuccess, options = {}) {
  const actionId = state.nextActionId + 1;
  state.nextActionId = actionId;
  state.latestActionId = actionId;
  const requestInputVersion = state.inputVersion;
  setBusy(true);
  setStatus(options.status || 'Working');
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    if (requestInputVersion !== state.inputVersion) {
      if (actionId === state.latestActionId) setStatus('Inputs changed; run again');
      return false;
    }
    await onSuccess(payload);
    return true;
  } catch (error) {
    if (actionId === state.latestActionId) setStatus(error.message || String(error), 'error');
    return false;
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

async function uploadSelectedStyleTts2VoiceFile(event) {
  const [file] = event.target.files || [];
  if (!file) return;
  await uploadStyleTts2Voice(file, file.name);
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

async function uploadStyleTts2Voice(blob, filename) {
  setBusy(true);
  setStatus('Saving StyleTTS2 voice sample');
  try {
    const form = new FormData();
    form.append('file', blob, filename || 'voice-sample.wav');
    const response = await fetch('/api/styletts2/voices/upload', {
      method: 'POST',
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    state.pendingStyletts2Voice = payload.voice?.id || '';
    if (!state.pendingStyletts2Style) {
      state.pendingStyletts2Style = state.pendingStyletts2Voice;
    }
    await loadStyleTts2Voices();
    if (state.pendingStyletts2Voice && selectHasValue(elements['styletts2-voice'], state.pendingStyletts2Voice)) {
      elements['styletts2-voice'].value = state.pendingStyletts2Voice;
    }
    if (state.pendingStyletts2Style && selectHasValue(elements['styletts2-style'], state.pendingStyletts2Style)) {
      elements['styletts2-style'].value = state.pendingStyletts2Style;
    }
    savePreferences();
    markActionInputsChanged();
    syncVoiceSelector();
    setStatus(`Saved StyleTTS2 voice sample ${payload.voice?.label || payload.voice?.id || ''}`.trim());
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  } finally {
    setBusy(false);
  }
}

async function toggleRecording(target) {
  if (state.recording && state.recordingTarget === target) {
    await stopRecording();
    return;
  }
  await startRecording(target);
}

async function startRecording(target) {
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
    state.recordingTarget = target;
    updateRecordingControls();
    setStatus(target === 'styletts2Voice' ? 'Recording StyleTTS2 voice sample' : 'Recording');
  } catch (error) {
    stopMediaCapture();
    if (state.audioContext) {
      await state.audioContext.close();
      state.audioContext = null;
    }
    state.recording = false;
    state.recordingTarget = '';
    updateRecordingControls();
    setStatus(error.message || String(error), 'error');
  }
}

async function stopRecording() {
  if (!state.recording) return;
  const target = state.recordingTarget;
  state.recording = false;
  state.recordingTarget = '';
  updateRecordingControls();
  stopMediaCapture();
  if (state.audioContext) {
    await state.audioContext.close();
    state.audioContext = null;
  }

  const wav = encodeWav(flattenChunks(state.recordingChunks), state.recordingSampleRate || 48000);
  state.recordingChunks = [];
  if (target === 'styletts2Voice') {
    await uploadStyleTts2Voice(wav, `voice-sample-${Date.now()}.wav`);
  } else {
    await uploadWav(wav, `recording-${Date.now()}.wav`);
  }
}

function stopMediaCapture() {
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
}

function updateRecordingControls() {
  const recordingAudio = state.recording && state.recordingTarget === 'audio';
  const recordingVoice = state.recording && state.recordingTarget === 'styletts2Voice';
  const voiceEnabled = elements.backend.value === 'styletts2';
  setRecordingState(elements.record, recordingAudio, 'Start recording audio');
  setRecordingState(elements['record-styletts2-voice'], recordingVoice, 'Start recording voice sample');
  elements.record.disabled = state.recording && !recordingAudio;
  elements['record-styletts2-voice'].disabled = (state.recording && !recordingVoice) || !voiceEnabled;
}

function setRecordingState(recordButton, isRecording, idleLabel) {
  setButtonIcon(
    recordButton,
    isRecording ? 'square' : 'circle',
    isRecording ? 'Stop recording' : idleLabel,
    isRecording ? 'Recording - stop recording' : 'Not recording - start recording',
    isRecording ? 'Stop' : 'Record',
  );
  recordButton.setAttribute('aria-pressed', isRecording ? 'true' : 'false');
  recordButton.dataset.state = isRecording ? 'recording' : 'idle';
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
  elements.phonemes.value = payload.phonemes || '';
  elements.syllables.value = formatSyllableTranscription(payload.syllables || []);
  elements.ir.textContent = JSON.stringify(payload.ir, null, 2);
  elements['phoneme-detail'].textContent = `${payload.variety}, ${countItems(payload.phonemes)} phonemes`;
  drawAll();
}

function formatSyllableTranscription(syllables) {
  const labels = syllables
    .map((syllable) => {
      const label = syllable.label || '';
      if (syllable.stress === 'primary') return `ˈ${label}`;
      if (syllable.stress === 'secondary') return `ˌ${label}`;
      return label;
    })
    .filter((label) => label.length > 0);
  const transcription = labels
    .map((label, index) => {
      if (index === 0 || label.startsWith('ˈ') || label.startsWith('ˌ')) return label;
      return `.${label}`;
    })
    .join('');

  return transcription ? `[${transcription}]` : '';
}

function renderAlignment(payload) {
  state.currentAlignment = payload;
  state.selection = null;
  state.playRangeEnd = null;
  elements.asr.value =
    payload.asr_transcript || (payload.asr_segments || []).map((segment) => segment.text).join(' ');
  elements['alignment-detail'].textContent = `${payload.words.length} words, ${payload.phonemes.length} phonemes, ${payload.phones.length} phones, ${(payload.feature_tracks || []).length} features, ${(payload.projected_voicing || []).length} projected, ${(payload.candidate_overlays || []).length} candidates`;
  drawAll();
}

function clearAlignment() {
  state.currentAlignment = null;
  state.selection = null;
  state.playRangeEnd = null;
  elements.asr.value = '';
  elements['alignment-detail'].textContent = 'No alignment';
  drawAll();
}

async function setAudio(url, detail) {
  state.currentAudioUrl = url;
  state.currentAlignment = null;
  state.selection = null;
  state.playRangeEnd = null;
  state.audioSeeking = false;
  elements.audio.src = url;
  elements.audio.load();
  elements['audio-detail'].textContent = detail || 'Audio ready';
  updateAudioTransport();
  updateActionButtons();
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
    updateAudioTransport();
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
  updateAudioTransport();
}

async function toggleAudioPlayback() {
  if (!elements.audio.src) {
    setStatus('Load or synthesize audio before playback', 'error');
    return;
  }
  if (!elements.audio.paused && !elements.audio.ended) {
    elements.audio.pause();
    return;
  }

  state.playRangeEnd = null;
  const duration = audioDuration();
  if (duration > 0 && elements.audio.currentTime >= duration - 0.01) {
    elements.audio.currentTime = 0;
  }
  try {
    await elements.audio.play();
  } catch (error) {
    setStatus(error.message || String(error), 'error');
  }
}

function onAudioProgressPointerDown(event) {
  if (!elements.audio.src) return;
  event.preventDefault();
  state.audioSeeking = true;
  elements['audio-progress'].setPointerCapture?.(event.pointerId);
  seekAudioFromProgressEvent(event);
}

function onAudioProgressPointerMove(event) {
  if (!state.audioSeeking) return;
  event.preventDefault();
  seekAudioFromProgressEvent(event);
}

function onAudioProgressPointerUp(event) {
  if (!state.audioSeeking) return;
  event.preventDefault();
  state.audioSeeking = false;
  elements['audio-progress'].releasePointerCapture?.(event.pointerId);
}

function onAudioProgressKeyDown(event) {
  if (!elements.audio.src) return;
  const duration = audioDuration();
  if (duration <= 0) return;
  const smallStep = event.shiftKey ? 0.25 : 1;
  const largeStep = Math.max(1, duration * 0.1);
  let nextTime = elements.audio.currentTime || 0;
  if (event.key === 'ArrowLeft') nextTime -= smallStep;
  else if (event.key === 'ArrowRight') nextTime += smallStep;
  else if (event.key === 'PageDown') nextTime -= largeStep;
  else if (event.key === 'PageUp') nextTime += largeStep;
  else if (event.key === 'Home') nextTime = 0;
  else if (event.key === 'End') nextTime = duration;
  else return;

  event.preventDefault();
  seekAudioToTime(nextTime);
}

function seekAudioFromProgressEvent(event) {
  const rect = elements['audio-progress'].getBoundingClientRect();
  const fraction = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
  seekAudioToTime(fraction * audioDuration());
}

function seekAudioToTime(time) {
  const duration = audioDuration();
  if (duration <= 0) return;
  state.playRangeEnd = null;
  elements.audio.currentTime = clamp(time, 0, duration);
  updateAudioTransport();
  drawAll();
}

function updateAudioTransport() {
  const duration = audioDuration();
  const current = duration > 0 ? clamp(elements.audio.currentTime || 0, 0, duration) : 0;
  const percent = duration > 0 ? (current / duration) * 100 : 0;

  elements['audio-position'].textContent = formatSeconds(current);
  elements['audio-duration'].textContent = formatSeconds(duration);
  elements['audio-progress-fill'].style.width = `${percent}%`;
  elements['audio-progress-thumb'].style.left = `${percent}%`;
  elements['audio-progress'].setAttribute('aria-valuenow', Math.round(percent).toString());
  elements['audio-progress'].setAttribute('aria-valuetext', `${formatSeconds(current)} of ${formatSeconds(duration)}`);
  const isPlaying = !elements.audio.paused && !elements.audio.ended;
  setButtonIcon(
    elements['audio-play-toggle'],
    isPlaying ? 'pause' : 'play',
    isPlaying ? 'Playing - pause audio' : 'Paused - play audio',
  );
  elements['audio-play-toggle'].setAttribute('aria-pressed', isPlaying ? 'true' : 'false');
  elements['audio-play-toggle'].dataset.state = isPlaying ? 'playing' : 'paused';
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
  const centerTime = xToTime(elements.timeline.scrollLeft + x);
  state.zoom = clamp(state.zoom * multiplier, 1, 240);
  const newDuration = viewDuration();
  const fraction = x / timelineViewportWidth();
  state.viewStart = clamp(centerTime - newDuration * fraction, 0, maxViewStart());
  if (Math.abs(oldDuration - newDuration) > 0.0001) drawAll();
}

function onTimelineWheel(event) {
  event.preventDefault();
  const rect = elements.timeline.getBoundingClientRect();
  const multiplier = event.deltaY < 0 ? 1.18 : 1 / 1.18;
  zoomAt(event.clientX - rect.left, multiplier);
}

function onTimelineScroll() {
  const maxScrollLeft = Math.max(0, elements.timeline.scrollWidth - elements.timeline.clientWidth);
  const maxStart = maxViewStart();
  const nextViewStart = maxScrollLeft > 0 ? (elements.timeline.scrollLeft / maxScrollLeft) * maxStart : 0;
  const clampedViewStart = clamp(nextViewStart, 0, maxStart);
  if (Math.abs(state.viewStart - clampedViewStart) < 0.0005) return;
  state.viewStart = clampedViewStart;
  drawAll({ syncScrollPosition: false });
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
    const secondsPerPixel = state.duration / contentWidth();
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
    updateAudioTransport();
    drawAll();
    if (!elements.audio.paused && !elements.audio.ended) {
      state.animationFrame = requestAnimationFrame(tick);
    }
  };
  tick();
}

function drawAll(options = {}) {
  syncTimelineContent(options);
  setupCanvases();
  drawRuler();
  drawWaveform();
  drawSpectrogram();
  drawTrack(canvases['feature-track'], state.currentAlignment?.feature_tracks || [], {
    kind: 'feature',
    color: featureTrackColor,
    text: '#f3f6f1',
    empty: 'voicing',
    label: voicingTrackLabel,
    centeredLabel: true,
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
    candidateSummary: true,
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
  const cssWidth = contentWidth();
  for (const canvas of Object.values(canvases)) {
    canvas.style.width = `${cssWidth}px`;
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

function syncTimelineContent({ syncScrollPosition = true } = {}) {
  const content = elements['timeline-content'];
  if (!content) return;

  const viewportWidth = timelineViewportWidth();
  const width = contentWidth();
  content.style.width = `${width}px`;
  for (const canvas of Object.values(canvases)) {
    canvas.style.width = `${width}px`;
  }

  if (!syncScrollPosition) return;

  const maxScrollLeft = Math.max(0, width - viewportWidth);
  const maxStart = maxViewStart();
  const nextScrollLeft = maxStart > 0 ? (state.viewStart / maxStart) * maxScrollLeft : 0;
  if (Math.abs(elements.timeline.scrollLeft - nextScrollLeft) > 0.5) {
    elements.timeline.scrollLeft = nextScrollLeft;
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
  const firstTick = 0;
  ctx.strokeStyle = '#293238';
  ctx.lineWidth = 1;
  for (let time = firstTick; time <= state.duration; time += seconds) {
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
  for (let time = firstTick; time <= state.duration; time += seconds) {
    const x = timeToX(time);
    ctx.beginPath();
    ctx.moveTo(x, height - 10);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.fillText(formatSeconds(time), x + 4, 14);
  }
  drawLaneCaption(ctx, width, 'Time');
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
  const samplesPerPixel = Math.max(1, Math.floor(samples.length / width));

  ctx.strokeStyle = '#6fd2a4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const start = x * samplesPerPixel;
    const end = Math.min(samples.length, start + samplesPerPixel);
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

  ctx.drawImage(state.spectrogramCanvas, 0, 0, width, height);

  drawGrid(ctx, width, height);
  drawSpectrogramCandidateOverlays(ctx, width, height);
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

  if (options.candidateSummary) {
    drawCandidateSummaryTrack(ctx, width, height, segments, options);
    drawLaneCaption(ctx, width, options.empty);
    return;
  }

  if (options.blendedChips) {
    drawBlendedTrack(ctx, width, height, segments, options);
    drawLaneCaption(ctx, width, options.empty);
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
    ctx.fillStyle = selected ? '#f3f6f1' : fillColor;
    ctx.globalAlpha = alpha;
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = selected ? '#ef8c86' : '#0b0e10';
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = 1;
    if (w > 16) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y, Math.max(0, w - 4), h);
      ctx.clip();
      const label = trackSegmentLabel(segment, options);
      ctx.fillStyle = selected ? '#0b0e10' : options.text;
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
      if (options.centeredLabel) {
        ctx.textAlign = 'center';
        ctx.fillText(label, x + w / 2, Math.floor(height / 2) + 4);
      } else {
        ctx.fillText(label, x + 5, Math.floor(height / 2) + 4);
      }
      ctx.restore();
    }
  }
  drawLaneCaption(ctx, width, options.empty);
}

function drawLaneCaption(ctx, width, label) {
  if (!label) return;
  ctx.save();
  ctx.fillStyle = 'rgba(155, 166, 161, 0.7)';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(String(label).toUpperCase(), width - 8, 7);
  ctx.restore();
}

function trackSegmentLabel(segment, options) {
  if (typeof options.label === 'function') return options.label(segment);
  return segment.label || segment.text || '';
}

function drawBlendedTrack(ctx, width, height, segments, options) {
  const y = 8;
  const h = height - 16;
  const visibleSegments = segments.flatMap((segment) => {
    const x = timeToX(segment.start_ms / 1000);
    const right = timeToX(segment.end_ms / 1000);
    const w = Math.max(1, right - x);
    if (right < 0 || x > width) return [];
    return [
      {
        segment,
        selected: isSelectedSegment(options.kind, segment),
        x,
        right,
        w,
        fillColor: typeof options.color === 'function' ? options.color(segment) : options.color,
        alpha: typeof options.alpha === 'function' ? options.alpha(segment) : 0.92,
      },
    ];
  });

  for (const item of visibleSegments) {
    const bleed = Math.min(10, Math.max(2, item.w * 0.16));
    const gradient = ctx.createLinearGradient(item.x - bleed, 0, item.right + bleed, 0);
    const chipColor = item.selected ? '#f3f6f1' : item.fillColor;
    gradient.addColorStop(0, rgba(chipColor, 0));
    gradient.addColorStop(0.18, rgba(chipColor, item.alpha * 0.76));
    gradient.addColorStop(0.5, rgba(chipColor, item.alpha));
    gradient.addColorStop(0.82, rgba(chipColor, item.alpha * 0.76));
    gradient.addColorStop(1, rgba(chipColor, 0));
    ctx.fillStyle = gradient;
    ctx.fillRect(item.x - bleed, y, item.w + bleed * 2, h);
  }

  for (const item of visibleSegments) {
    if (!item.selected) continue;
    ctx.strokeStyle = '#ef8c86';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(item.x + 1, y + 1, Math.max(1, item.w - 2), Math.max(1, h - 2));
    ctx.lineWidth = 1;
  }

  for (const item of visibleSegments) {
    if (item.w <= 16) continue;
    ctx.save();
    ctx.beginPath();
    ctx.rect(item.x + 2, y, Math.max(0, item.w - 4), h);
    ctx.clip();
    ctx.fillStyle = item.selected ? '#0b0e10' : options.text;
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(trackSegmentLabel(item.segment, options), item.x + item.w / 2, Math.floor(height / 2) + 4);
    ctx.restore();
  }
}

function drawCandidateSummaryTrack(ctx, width, height, segments, options) {
  const items = layoutCandidateTokenItems(
    ctx,
    visibleCandidateOverlayItems(segments, width, options).filter((item) => candidateDisplayLabel(item.segment)),
    width,
    height,
  );
  if (!items.length) {
    ctx.fillStyle = '#66727a';
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText(options.empty, 14, Math.floor(height / 2) + 4);
    return;
  }

  const baseline = Math.floor(height / 2) + 0.5;
  ctx.strokeStyle = 'rgba(102, 114, 122, 0.65)';
  ctx.beginPath();
  ctx.moveTo(0, baseline);
  ctx.lineTo(width, baseline);
  ctx.stroke();

  for (const item of items) {
    const chipColor = item.selected ? '#f3f6f1' : item.fillColor;
    const markerY = Math.max(4, item.labelY + item.labelHeight - 4);
    ctx.fillStyle = rgba(chipColor, item.selected ? 0.9 : item.alpha * 0.55);
    ctx.fillRect(item.x, markerY, item.w, 3);

    ctx.fillStyle = item.selected ? '#f3f6f1' : '#14191c';
    ctx.fillRect(item.labelX, item.labelY, item.labelWidth, item.labelHeight);
    ctx.strokeStyle = item.selected ? '#ef8c86' : rgba(chipColor, item.alpha * 0.86);
    ctx.lineWidth = item.selected ? 1.5 : 1;
    ctx.strokeRect(
      item.labelX + 0.5,
      item.labelY + 0.5,
      Math.max(1, item.labelWidth - 1),
      Math.max(1, item.labelHeight - 1),
    );
    ctx.lineWidth = 1;

    ctx.fillStyle = item.selected ? '#0b0e10' : '#f3f6f1';
    ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.label, item.labelX + item.labelWidth / 2, item.labelY + item.labelHeight / 2 + 0.5);
  }
}

function layoutCandidateTokenItems(ctx, items, width, height) {
  if (!items.length) return [];
  ctx.save();
  ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
  const laneCount = height >= 54 ? 2 : 1;
  const laneHeight = Math.floor((height - 8) / laneCount);
  const laneEnds = new Array(laneCount).fill(Number.NEGATIVE_INFINITY);
  for (const item of items) {
    item.label = candidateDisplayLabel(item.segment);
    item.labelWidth = Math.ceil(Math.max(22, Math.min(94, ctx.measureText(item.label).width + 12)));
    item.labelHeight = Math.max(18, Math.min(22, laneHeight - 3));
    item.labelX = clamp(item.x + item.w / 2 - item.labelWidth / 2, 3, Math.max(3, width - item.labelWidth - 3));
    const lane = firstOpenCandidateLane(laneEnds, item.labelX, item.labelWidth);
    if (lane === -1) {
      item.labelWidth = Math.min(item.labelWidth, Math.max(18, item.w + 8));
      item.label = fitCanvasText(ctx, item.label, item.labelWidth - 6);
      item.labelX = clamp(item.x + item.w / 2 - item.labelWidth / 2, 3, Math.max(3, width - item.labelWidth - 3));
      item.lane = 0;
    } else {
      item.lane = lane;
    }
    item.labelY = 4 + item.lane * laneHeight + Math.floor((laneHeight - item.labelHeight) / 2);
    laneEnds[item.lane] = Math.max(laneEnds[item.lane], item.labelX + item.labelWidth + 4);
  }
  ctx.restore();
  return items.filter((item) => item.label);
}

function firstOpenCandidateLane(laneEnds, labelX, labelWidth) {
  for (let lane = 0; lane < laneEnds.length; lane += 1) {
    if (laneEnds[lane] <= labelX) return lane;
  }
  return -1;
}

function candidateDisplayLabel(segment) {
  return candidateTokenLabel(segment) || candidateFeatureLabel(segment);
}

function candidateTokenLabel(segment) {
  const tokenId = String(segment.token_id || '');
  if (!tokenId) return '';

  const label = String(segment.label || '').trim();
  if (label) {
    const compact = label.includes(':') ? label.slice(label.lastIndexOf(':') + 1).trim() : label;
    if (compact && !/\s/.test(compact)) return compact;
  }

  const parts = tokenId.split('.');
  return parts[parts.length - 1] || tokenId;
}

function candidateFeatureLabel(segment) {
  const kind = String(segment.kind || '');
  const label = String(segment.label || '').toLowerCase();
  if (kind === 'periodic_voicing' || label.includes('voicing')) return '[+voice]';
  if (kind === 'nucleus_candidate' || label.includes('vowel')) return '[+syll]';
  if (kind === 'sonority_peak') return '[+son]';
  if (kind === 'frication_noise' || label.includes('fric') || label.includes('centroid')) return '[+fric]';
  if (kind === 'sibilant_noise' || label.includes('sibil') || label.includes('skew')) return '[+strid]';
  if (kind === 'stop_closure' || label.includes('closure')) return '[-cont]';
  if (kind === 'release_burst' || label.includes('release') || label.includes('burst')) return '[rel]';
  if (kind === 'aspiration') return '[+spread]';
  if (kind === 'boundary' || label.includes('silence')) return '[#]';
  if (kind === 'rhotic_region' || label.includes('rhotic')) return '[+rhotic]';
  if (
    kind === 'nasal_murmur' ||
    kind === 'nasal_antiresonance' ||
    kind === 'nasal_place' ||
    label.includes('nasal')
  ) {
    return '[+nasal]';
  }
  if (kind === 'approximant_formants') return '[+approx]';
  if (kind === 'tap_closure') return '[tap]';
  if (kind === 'formant_region' && /^f[123]$/.test(label)) return `[${label.toUpperCase()}]`;
  if (kind === 'formant_trajectory' || label.includes('formant')) return '[form]';
  return '';
}

function drawSpectrogramCandidateOverlays(ctx, width, height) {
  const segments = state.currentAlignment?.candidate_overlays || [];
  if (!segments.length) return;

  const layout = layoutSpectrogramCandidateOverlays(ctx, segments, width, height);
  for (const item of layout) {
    const color = candidateOverlayColor(item.segment);
    const alpha = candidateOverlayAlpha(item.segment);
    ctx.fillStyle = rgba(color, alpha * 0.1);
    ctx.fillRect(item.x, item.y - 6, item.w, 12);
    ctx.strokeStyle = rgba(color, alpha * 0.42);
    ctx.lineWidth = item.selected ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(item.x, item.y + 0.5);
    ctx.lineTo(item.right, item.y + 0.5);
    ctx.stroke();
    ctx.lineWidth = 1;
  }
}

function layoutSpectrogramCandidateOverlays(ctx, segments, width, height) {
  const items = visibleCandidateOverlayItems(segments, width, {
    kind: 'candidate_overlay',
    color: candidateOverlayColor,
    alpha: candidateOverlayAlpha,
  });
  for (const item of items) {
    item.y = spectrogramCandidateY(item.segment, height);
    item.showLabel = false;
  }
  return items;
}

function visibleCandidateOverlayItems(segments, width, options) {
  return segments
    .flatMap((segment) => {
      const x = timeToX(segment.start_ms / 1000);
      const right = timeToX(segment.end_ms / 1000);
      const w = Math.max(1, right - x);
      if (right < 0 || x > width) return [];
      return [
        {
          segment,
          selected: isSelectedSegment(options.kind, segment),
          x,
          right,
          w,
          fillColor: typeof options.color === 'function' ? options.color(segment) : options.color,
          alpha: typeof options.alpha === 'function' ? options.alpha(segment) : 0.92,
          lane: 0,
          y: 0,
          h: 0,
        },
      ];
    })
    .sort((left, right) => {
      return (
        left.x - right.x ||
        left.right - right.right ||
        candidateOverlaySourceOrder(left.segment) - candidateOverlaySourceOrder(right.segment) ||
        String(left.segment.kind || '').localeCompare(String(right.segment.kind || ''))
      );
    });
}

function spectrogramCandidateY(segment, height) {
  const band = spectrogramCandidateBandKey(segment);
  const ratios = {
    high_noise: 0.24,
    release: 0.36,
    measurement: 0.5,
    nucleus: 0.62,
    reverse: 0.76,
    inferred: 0.14,
  };
  return Math.round(clamp(ratios[band] ?? 0.46, 0.08, 0.88) * height);
}

function spectrogramCandidateBandKey(segment) {
  const source = String(segment.source || '');
  const kind = String(segment.kind || '').toLowerCase();
  const label = String(segment.label || '').toLowerCase();
  const text = `${kind} ${label}`;
  if (source === 'syllable_nucleus' || text.includes('nucleus') || text.includes('vowel')) return 'nucleus';
  if (source === 'reverse_snipper') return 'reverse';
  if (source === 'inference_rule') return 'inferred';
  if (source === 'acoustic_measurement') return 'measurement';
  if (
    text.includes('fric') ||
    text.includes('sibil') ||
    text.includes('strident') ||
    text.includes('noise')
  ) {
    return 'high_noise';
  }
  if (text.includes('burst') || text.includes('release') || text.includes('onset')) return 'release';
  if (source === 'acoustic_landmark') return 'release';
  return 'measurement';
}

function candidateOverlaySourceOrder(segment) {
  const order = {
    syllable_nucleus: 0,
    reverse_snipper: 1,
    acoustic_cue: 2,
    acoustic_landmark: 3,
    acoustic_measurement: 4,
    inference_rule: 5,
  };
  return order[segment.source] ?? 99;
}

function fitCanvasText(ctx, text, maxWidth) {
  if (maxWidth <= 6) return '';
  if (ctx.measureText(text).width <= maxWidth) return text;
  const ellipsis = '...';
  let left = 0;
  let right = text.length;
  while (left < right) {
    const mid = Math.ceil((left + right) / 2);
    if (ctx.measureText(`${text.slice(0, mid)}${ellipsis}`).width <= maxWidth) {
      left = mid;
    } else {
      right = mid - 1;
    }
  }
  return left > 0 ? `${text.slice(0, left)}${ellipsis}` : '';
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

function voicingTrackLabel(segment) {
  return segment.kind === 'voiced' ? '+' : '-';
}

function projectedVoicingColor(segment) {
  if (segment.kind === 'voiced') return '#2f9cc9';
  if (segment.kind === 'unvoiced') return '#9f7fd2';
  return '#66727a';
}

function candidateOverlayColor(segment) {
  if (segment.source === 'reverse_snipper') return '#e8c36f';
  if (segment.source === 'syllable_nucleus') return '#ef8c86';
  if (segment.source === 'acoustic_cue') return '#79b8ff';
  if (segment.source === 'acoustic_landmark') return '#6fd2a4';
  if (segment.source === 'acoustic_measurement') return '#c3a4ff';
  if (segment.source === 'inference_rule') return '#f3f6f1';
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
  if (kind === 'spectrogram_candidate') {
    return hitTestSpectrogramCandidateOverlay(event, alignment, timeMs);
  }
  if (kind === 'candidate_overlay') {
    return hitTestCandidateOverlay(event, alignment, timeMs);
  }
  const segments = alignmentSegments(alignment, kind);
  const segment = segments.find((candidate) => {
    return timeMs >= candidate.start_ms && timeMs <= candidate.end_ms;
  });
  return segment ? { kind, segment } : null;
}

function hitTestCandidateOverlay(event, alignment, timeMs) {
  const segment = [...(alignment.candidate_overlays || [])]
    .filter((candidate) => timeMs >= candidate.start_ms && timeMs <= candidate.end_ms)
    .sort((left, right) => Number(right.confidence || 0) - Number(left.confidence || 0))[0];
  return segment ? { kind: 'candidate_overlay', segment } : null;
}

function hitTestSpectrogramCandidateOverlay(event, alignment, timeMs) {
  const canvas = event.target;
  const rect = canvas.getBoundingClientRect();
  const y = event.clientY - rect.top;
  const layout = layoutSpectrogramCandidateOverlays(
    canvas.getContext('2d'),
    alignment.candidate_overlays || [],
    canvas.clientWidth,
    canvas.clientHeight,
  );
  const segment = [...layout].reverse().find((item) => {
    const onGuide = timeMs >= item.segment.start_ms && timeMs <= item.segment.end_ms && Math.abs(y - item.y) <= 7;
    const onLabel =
      item.showLabel &&
      event.clientX - rect.left >= item.hitLeft &&
      event.clientX - rect.left <= item.hitRight &&
      y >= item.hitTop &&
      y <= item.hitBottom;
    return onGuide || onLabel;
  })?.segment;
  return segment ? { kind: 'candidate_overlay', segment } : null;
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
  if (target.id === 'spectrogram') return 'spectrogram_candidate';
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
  return clamp(event.clientX - rect.left + elements.timeline.scrollLeft, 0, contentWidth());
}

function formatRange(start, end) {
  return `${formatSeconds(start)}-${formatSeconds(end)}`;
}

function toggleIr() {
  state.irVisible = !state.irVisible;
  elements.ir.hidden = !state.irVisible;
  setButtonIcon(
    elements['toggle-ir'],
    state.irVisible ? 'chevron-down' : 'chevron-right',
    state.irVisible ? 'Hide Speech IR' : 'Show Speech IR',
  );
}

function setBusy(busy) {
  state.activeActionCount = Math.max(0, state.activeActionCount + (busy ? 1 : -1));
  if (busy) state.inputsChangedDuringAction = false;
  updateActionButtons();
}

function markActionInputsChanged() {
  state.inputVersion += 1;
  if (state.activeActionCount > 0) {
    state.inputsChangedDuringAction = true;
  }
  updateActionButtons();
}

function updateActionButtons() {
  const busyDisabled = state.activeActionCount > 0 && !state.inputsChangedDuringAction;
  elements.phonemicize.disabled = busyDisabled;
  elements.synthesize.disabled = busyDisabled;
  elements.align.disabled = state.alignInProgress || busyDisabled || !state.currentAudioUrl;
}

function setAlignProgress(inProgress) {
  state.alignInProgress = inProgress;
  elements.align.textContent = inProgress ? 'Aligning...' : 'Align ASR';
  if (inProgress) {
    elements.align.setAttribute('aria-busy', 'true');
    elements.align.dataset.progress = 'true';
  } else {
    elements.align.removeAttribute('aria-busy');
    delete elements.align.dataset.progress;
  }
  updateActionButtons();
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

function timelineViewportWidth() {
  const rectWidth = Math.floor(elements.timeline.getBoundingClientRect().width);
  const parentWidth = elements.timeline.parentElement?.clientWidth || 0;
  return Math.max(1, rectWidth, elements.timeline.clientWidth, parentWidth);
}

function contentWidth() {
  return Math.max(timelineViewportWidth(), Math.round(timelineViewportWidth() * state.zoom));
}

function audioDuration() {
  if (Number.isFinite(elements.audio.duration) && elements.audio.duration > 0) {
    return elements.audio.duration;
  }
  if (state.audioBuffer?.duration > 0) return state.audioBuffer.duration;
  return 0;
}

function maxViewStart() {
  return Math.max(0, state.duration - viewDuration());
}

function timeToX(time) {
  return (time / Math.max(0.001, state.duration)) * contentWidth();
}

function xToTime(x) {
  return (x / contentWidth()) * state.duration;
}

function tickSeconds() {
  const targetTicks = Math.max(4, timelineViewportWidth() / 120);
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
