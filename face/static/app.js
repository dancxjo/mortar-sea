window.faceApp = function faceApp() {
  return {
    cameraMessage: 'Camera idle',
    clientId: 'face-browser',
    sensorId: 'camera.default',
    locationSensorId: 'gps.default',
    vision: {
      socket: null,
      status: 'disconnected',
      pending: false,
      sent: 0,
      acked: 0,
      dropped: 0,
      lastError: '',
    },
    location: {
      socket: null,
      status: 'disconnected',
      pending: false,
      sent: 0,
      acked: 0,
      dropped: 0,
      lastError: '',
      lastFix: '',
      pendingPosition: null,
      watchId: null,
    },
    asr: {
      socket: null,
      status: 'disconnected',
      sent: 0,
      acked: 0,
      dropped: 0,
      lastError: '',
      lastTranscript: '',
      sampleRate: 0,
      clipMs: 500,
      chunks: [],
      queuedSamples: 0,
    },
    fps: 3,
    experiencePrompt: '',
    experienceResponse: '',
    experienceSocket: null,
    experienceStatus: 'disconnected',
    activeExperienceGenerationId: null,
    activeVoiceGenerationId: null,
    faceEmoji: '🤔',
    voiceResponse: '',
    voiceHasTokens: false,
    voiceStatus: 'waiting',
    voiceLastError: '',
    voicePlaybackDetail: '',
    voicePlaybackEvents: {
      drafts: 0,
      tts: 0,
      audio: 0,
      started: 0,
      finished: 0,
      interrupted: 0,
    },
    voiceAudio: null,
    voiceAudioUrl: null,
    voiceAudioContext: null,
    voiceAudioSource: null,
    voiceCurrentDraft: null,
    voiceUtteranceStartedAt: null,
    voiceMouthOpen: false,
    voiceAwaitingServerAudioTimer: null,
    voiceActiveSpeechUtterance: null,
    llmJobs: [],
    selectedLlmJobId: null,
    mime: 'image/jpeg',
    quality: 0.45,
    running: false,
    sequence: 0,
    locationSequence: 0,
    asrSequence: 0,
    stream: null,
    audioContext: null,
    audioSource: null,
    audioProcessor: null,
    audioSink: null,
    targetWidth: 160,
    timer: null,

    async init() {
      this.connectRealtimeExperience();
    },

    voicePlaybackEventSummary() {
      return `draft ${this.voicePlaybackEvents.drafts} / tts ${this.voicePlaybackEvents.tts} / audio ${this.voicePlaybackEvents.audio} / started ${this.voicePlaybackEvents.started} / finished ${this.voicePlaybackEvents.finished} / interrupted ${this.voicePlaybackEvents.interrupted}`;
    },

    async start() {
      this.cameraMessage = 'Requesting camera';
      try {
        await this.unlockVoicePlayback();
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: this.targetWidth },
            facingMode: 'user',
          },
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        this.$refs.video.srcObject = this.stream;
        await this.$refs.video.play();
        this.running = true;
        this.cameraMessage = 'Camera running';
        this.syncVision();
        this.startLocation();
        await this.startAsr();
        this.scheduleCapture();
      } catch (error) {
        this.cameraMessage = error.message || 'Camera permission failed';
      }
    },

    stop() {
      this.running = false;
      window.clearTimeout(this.timer);
      this.stopVoiceMouth('face stopped');
      this.disconnectVision();
      this.stopLocation();
      this.stopAsr();
      this.stopVoicePlaybackContext();
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      this.cameraMessage = 'Camera idle';
    },

    stopVoiceMouth(reason) {
      this.clearVoiceAudioFallbackTimer();
      this.cancelBrowserSpeechFallback();
      const interruptedDraft = this.voiceCurrentDraft;
      this.voiceCurrentDraft = null;
      this.voiceUtteranceStartedAt = null;
      if (interruptedDraft) {
        this.sendVoiceMouthEvent('voice_speech_interrupted', interruptedDraft, { reason });
      }
      if (this.voiceAudio) {
        this.voiceAudio.pause();
        this.voiceAudio.src = '';
        this.voiceAudio = null;
      }
      if (this.voiceAudioUrl) {
        URL.revokeObjectURL(this.voiceAudioUrl);
        this.voiceAudioUrl = null;
      }
      if (this.voiceAudioSource) {
        try {
          this.voiceAudioSource.stop();
        } catch (_) {
          // The source may already have ended.
        }
        this.voiceAudioSource.disconnect();
        this.voiceAudioSource = null;
      }
      this.voiceMouthOpen = false;
    },

    async unlockVoicePlayback() {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        this.voiceLastError = 'Web Audio unavailable';
        this.voicePlaybackDetail = this.voiceLastError;
        return;
      }
      if (!this.voiceAudioContext) {
        this.voiceAudioContext = new AudioContextClass();
      }
      if (this.voiceAudioContext.state !== 'running') {
        await this.voiceAudioContext.resume();
      }
      this.voicePlaybackDetail = `Voice audio context ${this.voiceAudioContext.state} at ${Math.round(this.voiceAudioContext.sampleRate)} Hz`;
    },

    stopVoicePlaybackContext() {
      if (this.voiceAudioSource) {
        try {
          this.voiceAudioSource.stop();
        } catch (_) {
          // The source may already have ended.
        }
        this.voiceAudioSource.disconnect();
        this.voiceAudioSource = null;
      }
      if (this.voiceAudioContext) {
        this.voiceAudioContext.close();
        this.voiceAudioContext = null;
      }
    },

    connectRealtimeExperience() {
      if (this.experienceSocket && this.experienceSocket.readyState <= WebSocket.OPEN) return;

      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/realtime-experience`);
      this.experienceSocket = socket;
      this.experienceStatus = 'connecting';

      socket.addEventListener('open', () => {
        this.experienceStatus = 'connected';
      });

      socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (this.isLlmJobEvent(message.type)) {
          this.upsertLlmJob(message);
          return;
        }
        if (message.type === 'voice_response_start') {
          if (this.voiceResponse && this.activeVoiceGenerationId !== message.generation_id) {
            this.voiceResponse += '\n';
            this.scrollVoiceStream();
          }
          this.activeVoiceGenerationId = message.generation_id;
          this.voiceHasTokens = Boolean(this.voiceResponse);
          this.voiceStatus = 'thinking';
          return;
        }
        if (message.type === 'voice_response_token') {
          if (message.generation_id !== this.activeVoiceGenerationId) {
            if (this.voiceResponse) {
              this.voiceResponse += '\n';
            }
            this.activeVoiceGenerationId = message.generation_id;
            this.voiceHasTokens = Boolean(this.voiceResponse);
            this.voiceStatus = 'thinking';
          }
          if (!this.voiceHasTokens) {
            this.voiceResponse = '';
            this.voiceHasTokens = true;
          }
          this.voiceResponse += message.text;
          this.scrollVoiceStream();
          return;
        }
        if (message.type === 'voice_response_done') {
          if (message.generation_id !== this.activeVoiceGenerationId) return;
          if (this.voiceStatus !== 'speaking') {
            this.voiceStatus = 'waiting';
          }
          return;
        }
        if (message.type === 'voice_speech_draft') {
          this.activeVoiceGenerationId = message.generation_id;
          this.voicePlaybackEvents.drafts += 1;
          this.prepareVoiceDraft(message);
          return;
        }
        if (message.type === 'voice_speech_synthesis_started') {
          this.activeVoiceGenerationId = message.generation_id;
          this.voicePlaybackEvents.tts += 1;
          this.voicePlaybackDetail = `Server TTS started for "${message.text || ''}"`;
          if (this.voiceCurrentDraft && this.voiceDraftMatches(this.voiceCurrentDraft, message)) {
            this.armBrowserSpeechFallback(this.voiceCurrentDraft);
          }
          return;
        }
        if (message.type === 'voice_speech_audio') {
          this.activeVoiceGenerationId = message.generation_id;
          this.voicePlaybackEvents.audio += 1;
          this.clearVoiceAudioFallbackTimer();
          this.cancelBrowserSpeechFallback();
          this.playVoiceSpeechAudio(message);
          return;
        }
        if (message.type === 'voice_speech_started') {
          if (message.generation_id !== this.activeVoiceGenerationId) return;
          this.voiceStatus = 'speaking';
          this.voicePlaybackEvents.started += 1;
          return;
        }
        if (message.type === 'voice_speech_finished') {
          if (message.generation_id !== this.activeVoiceGenerationId) return;
          this.voiceStatus = 'thinking';
          this.voicePlaybackEvents.finished += 1;
          return;
        }
        if (message.type === 'voice_speech_interrupted') {
          if (message.generation_id !== this.activeVoiceGenerationId) return;
          this.voiceStatus = 'thinking';
          this.voicePlaybackEvents.interrupted += 1;
          if (message.reason) {
            this.voiceLastError = message.reason;
            this.voicePlaybackDetail = message.reason;
          }
          return;
        }
        if (message.type === 'voice_observation') {
          this.activeVoiceGenerationId = message.generation_id;
          if (message.observation?.emoji) {
            this.faceEmoji = message.observation.emoji;
          }
          return;
        }
        if (message.type === 'face_emoji') {
          this.activeVoiceGenerationId = message.generation_id;
          if (message.emoji) {
            this.faceEmoji = message.emoji;
          }
          return;
        }
        if (message.type === 'asr_transcript') {
          this.asr.lastTranscript = message.text || '';
          return;
        }
        if (message.type === 'prompt') {
          this.activeExperienceGenerationId = message.generation_id;
          this.experiencePrompt = message.prompt;
          this.experienceResponse = '';
          return;
        }
        if (message.generation_id !== this.activeExperienceGenerationId) return;
        if (message.type === 'response_start') {
          this.experienceResponse = '';
          this.experienceStatus = 'generating';
          return;
        }
        if (message.type === 'response_token') {
          this.experienceResponse += message.text;
          return;
        }
        if (message.type === 'response_done') {
          this.experienceStatus = 'connected';
        }
      });

      socket.addEventListener('close', () => {
        this.experienceStatus = 'disconnected';
        window.setTimeout(() => this.connectRealtimeExperience(), 1000);
      });

      socket.addEventListener('error', () => {
        this.experienceStatus = 'error';
      });
    },

    scrollVoiceStream() {
      this.$nextTick(() => {
        const stream = this.$refs.voiceStream;
        if (!stream) return;
        stream.scrollTop = stream.scrollHeight;
      });
    },

    prepareVoiceDraft(draft) {
      this.stopVoiceMouth('superseded by newer voice draft');
      this.voiceCurrentDraft = draft;
      this.voiceStatus = 'synthesizing';
      this.voiceLastError = '';
      this.voicePlaybackDetail = `Waiting for server audio for "${draft.text || ''}"`;
      this.voiceMouthOpen = false;
    },

    clearVoiceAudioFallbackTimer() {
      if (this.voiceAwaitingServerAudioTimer) {
        window.clearTimeout(this.voiceAwaitingServerAudioTimer);
        this.voiceAwaitingServerAudioTimer = null;
      }
    },

    cancelBrowserSpeechFallback() {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      this.voiceActiveSpeechUtterance = null;
    },

    armBrowserSpeechFallback(draft) {
      this.clearVoiceAudioFallbackTimer();
      if (!draft?.text || !window.speechSynthesis || typeof window.SpeechSynthesisUtterance !== 'function') {
        return;
      }
      this.voiceAwaitingServerAudioTimer = window.setTimeout(() => {
        if (!this.voiceCurrentDraft || !this.voiceDraftMatches(this.voiceCurrentDraft, draft)) return;
        if (this.voiceAudioSource) return;

        const utterance = new window.SpeechSynthesisUtterance(draft.text);
        this.voiceActiveSpeechUtterance = utterance;
        this.voicePlaybackDetail = `Server audio timeout; browser fallback speaking "${draft.text}"`;

        utterance.onstart = () => {
          this.voiceStatus = 'speaking';
          this.voiceUtteranceStartedAt = performance.now();
          this.voiceMouthOpen = true;
          this.sendVoiceMouthEvent('voice_speech_started', draft);
        };
        utterance.onend = () => {
          const playbackDurationMs = this.voiceUtteranceStartedAt
            ? Math.max(0, Math.round(performance.now() - this.voiceUtteranceStartedAt))
            : null;
          this.voiceMouthOpen = false;
          this.sendVoiceMouthEvent('voice_speech_finished', draft, { duration_ms: playbackDurationMs });
          this.voiceActiveSpeechUtterance = null;
          this.clearFinishedVoiceDraft(draft);
        };
        utterance.onerror = (event) => {
          this.voiceMouthOpen = false;
          this.voiceActiveSpeechUtterance = null;
          const reason = event?.error
            ? `Browser speech fallback failed: ${event.error}`
            : 'Browser speech fallback failed';
          this.voiceLastError = reason;
          this.voicePlaybackDetail = reason;
          this.sendVoiceMouthEvent('voice_speech_interrupted', draft, { reason });
          this.clearFinishedVoiceDraft(draft);
        };
        window.speechSynthesis.speak(utterance);
      }, 2400);
    },

    async playVoiceSpeechAudio(audioMessage) {
      const draft = this.voiceCurrentDraft;
      if (!draft || !this.voiceDraftMatches(draft, audioMessage)) {
        return;
      }

      await this.unlockVoicePlayback();
      if (!this.voiceAudioContext || this.voiceAudioContext.state !== 'running') {
        const reason = `Voice audio context is ${this.voiceAudioContext?.state || 'unavailable'}`;
        this.voiceLastError = reason;
        this.voicePlaybackDetail = reason;
        this.sendVoiceMouthEvent('voice_speech_interrupted', draft, { reason });
        this.clearFinishedVoiceDraft(draft);
        return;
      }

      const encodedAudio = audioMessage.data || '';
      const audioBytes = this.base64ToArrayBuffer(encodedAudio);
      const durationMs = audioMessage.duration_ms;
      const samples = audioMessage.samples;
      this.voicePlaybackDetail = `Server audio ready: ${durationMs ?? '?'} ms, ${samples ?? '?'} samples, ${audioBytes.byteLength} bytes`;
      console.info('Mortar voice WAV ready', {
        utterance_id: draft.utterance_id,
        duration_ms: durationMs,
        samples,
        bytes: audioBytes.byteLength,
      });

      let audioBuffer;
      try {
        audioBuffer = await this.voiceAudioContext.decodeAudioData(audioBytes.slice(0));
      } catch (error) {
        const reason = error.message || 'Server WAV decode failed in browser';
        this.voiceLastError = reason;
        this.voicePlaybackDetail = reason;
        this.sendVoiceMouthEvent('voice_speech_interrupted', draft, { reason });
        this.clearFinishedVoiceDraft(draft);
        return;
      }

      if (this.voiceCurrentDraft !== draft) return;

      if (this.voiceAudioSource) {
        try {
          this.voiceAudioSource.stop();
        } catch (_) {
          // The source may already have ended.
        }
        this.voiceAudioSource.disconnect();
      }

      const source = this.voiceAudioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.voiceAudioContext.destination);
      this.voiceAudioSource = source;
      source.onended = () => {
        if (this.voiceAudioSource === source) {
          this.voiceAudioSource = null;
        }
        if (this.voiceCurrentDraft !== draft) return;
        const playbackDurationMs = this.voiceUtteranceStartedAt
          ? Math.max(0, Math.round(performance.now() - this.voiceUtteranceStartedAt))
          : null;
        this.voiceMouthOpen = false;
        this.voicePlaybackDetail = `Playback finished after ${playbackDurationMs} ms`;
        this.sendVoiceMouthEvent('voice_speech_finished', draft, { duration_ms: playbackDurationMs });
        this.clearFinishedVoiceDraft(draft);
      };

      try {
        source.start(0);
        this.voiceUtteranceStartedAt = performance.now();
        this.voiceMouthOpen = true;
        this.voiceStatus = 'speaking';
        this.voicePlaybackDetail = `Playing ${audioBuffer.duration.toFixed(2)}s through Web Audio`;
        console.info('Mortar voice playback started', {
          utterance_id: draft.utterance_id,
          duration: audioBuffer.duration,
          sample_rate: audioBuffer.sampleRate,
        });
        this.sendVoiceMouthEvent('voice_speech_started', draft);
      } catch (error) {
        this.voiceMouthOpen = false;
        this.voiceLastError = error.message || 'Web Audio playback failed';
        this.voicePlaybackDetail = this.voiceLastError;
        this.sendVoiceMouthEvent('voice_speech_interrupted', draft, {
          reason: this.voiceLastError,
        });
        this.clearFinishedVoiceDraft(draft);
      }
    },

    voiceDraftMatches(draft, message) {
      return draft
        && message
        && draft.utterance_id === message.utterance_id
        && draft.generation_id === message.generation_id;
    },

    base64ToArrayBuffer(data) {
      const binary = window.atob(data);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      return bytes.buffer;
    },

    clearFinishedVoiceDraft(draft) {
      if (this.voiceCurrentDraft !== draft) return;
      this.clearVoiceAudioFallbackTimer();
      this.voiceActiveSpeechUtterance = null;
      if (this.voiceAudio) {
        this.voiceAudio.pause();
        this.voiceAudio.src = '';
        this.voiceAudio = null;
      }
      if (this.voiceAudioUrl) {
        URL.revokeObjectURL(this.voiceAudioUrl);
        this.voiceAudioUrl = null;
      }
      this.voiceCurrentDraft = null;
      this.voiceUtteranceStartedAt = null;
      this.voiceMouthOpen = false;
    },

    sendVoiceMouthEvent(type, draft, extra = {}) {
      if (!this.experienceSocket || this.experienceSocket.readyState !== WebSocket.OPEN) return;
      this.experienceSocket.send(JSON.stringify({
        type,
        utterance_id: draft.utterance_id,
        generation_id: draft.generation_id,
        observed_at: new Date().toISOString(),
        text: draft.text || '',
        ...extra,
      }));
    },

    isLlmJobEvent(type) {
      return [
        'llm_job_queued',
        'llm_job_started',
        'llm_job_completed',
        'llm_job_failed',
      ].includes(type);
    },

    upsertLlmJob(message) {
      const phase = {
        llm_job_queued: 'queued',
        llm_job_started: 'running',
        llm_job_completed: 'completed',
        llm_job_failed: 'failed',
      }[message.type];
      const index = this.llmJobs.findIndex((job) => job.id === message.job_id);
      const existing = index >= 0 ? this.llmJobs[index] : {};
      const next = {
        ...existing,
        id: message.job_id,
        kind: message.job_kind || existing.kind || 'llm',
        phase,
        observedAt: message.observed_at,
      };

      if (message.type === 'llm_job_queued') {
        next.queuedAt = message.observed_at;
        next.priority = message.priority;
        next.messageCount = message.message_count;
        next.imageCount = message.image_count;
        next.promptChars = message.prompt_chars;
        next.maxTokens = message.max_tokens;
        next.stopCount = message.stop_count;
        next.promptPreview = message.prompt_preview;
      } else if (message.type === 'llm_job_started') {
        next.queueWaitMs = message.queue_wait_ms;
      } else if (message.type === 'llm_job_completed') {
        next.responseChars = message.response_chars;
        next.response = message.response;
        next.tokenEvents = message.token_events;
        next.elapsedMs = message.elapsed_ms;
      } else if (message.type === 'llm_job_failed') {
        next.error = message.error;
      }

      if (index >= 0) {
        this.llmJobs.splice(index, 1);
      }
      this.llmJobs.unshift(next);
      this.llmJobs = this.llmJobs.slice(0, 24);
      if (!this.selectedLlmJobId || !this.llmJobs.some((job) => job.id === this.selectedLlmJobId)) {
        this.selectedLlmJobId = next.id;
      }
    },

    visibleLlmJobs() {
      return this.llmJobs.slice(0, 14);
    },

    selectLlmJob(jobId) {
      this.selectedLlmJobId = jobId;
    },

    selectedLlmJob() {
      return this.llmJobs.find((job) => job.id === this.selectedLlmJobId) || this.llmJobs[0] || null;
    },

    llmActivityLabel() {
      const running = this.llmJobs.filter((job) => job.phase === 'running').length;
      const queued = this.llmJobs.filter((job) => job.phase === 'queued').length;
      if (running && queued) return `${running} running, ${queued} queued`;
      if (running) return `${running} running`;
      if (queued) return `${queued} queued`;
      if (this.llmJobs.length) return 'idle';
      return 'waiting';
    },

    llmActivityStatus() {
      if (this.llmJobs.some((job) => job.phase === 'running')) return 'running';
      if (this.llmJobs.some((job) => job.phase === 'queued')) return 'queued';
      if (this.llmJobs.some((job) => job.phase === 'failed')) return 'failed';
      return 'idle';
    },

    formatJobKind(kind) {
      return kind.replaceAll('_', ' ');
    },

    formatJobMetric(job) {
      if (job.phase === 'queued') {
        const tokenLabel = job.maxTokens ? `max ${job.maxTokens} tokens` : 'uncapped';
        return `${job.promptChars || 0} prompt chars, ${tokenLabel}`;
      }
      if (job.phase === 'running') {
        return `started after ${this.formatDuration(job.queueWaitMs || 0)}`;
      }
      if (job.phase === 'completed') {
        const empty = job.responseChars === 0 ? ', empty response' : '';
        return `${job.tokenEvents || 0} token events, ${job.responseChars || 0} chars${empty}, ${this.formatDuration(job.elapsedMs || 0)}`;
      }
      if (job.phase === 'failed') {
        return job.error || 'failed';
      }
      return '';
    },

    formatDuration(milliseconds) {
      if (milliseconds === undefined || milliseconds === null) return '-';
      if (milliseconds < 1000) return `${milliseconds}ms`;
      if (milliseconds < 10000) return `${(milliseconds / 1000).toFixed(1)}s`;
      return `${Math.round(milliseconds / 1000)}s`;
    },

    formatTimestamp(value) {
      if (!value) return '-';
      return new Intl.DateTimeFormat([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }).format(new Date(value));
    },

    formatCount(value, unit) {
      if (value === undefined || value === null) return '-';
      return `${value} ${unit}`;
    },

    scheduleCapture() {
      window.clearTimeout(this.timer);
      if (!this.running) return;
      const delay = Math.max(80, Math.round(1000 / this.fps));
      this.timer = window.setTimeout(async () => {
        await this.captureFrame();
        this.scheduleCapture();
      }, delay);
    },

    syncVision() {
      if (!this.running) {
        this.disconnectVision();
        return;
      }
      if (this.vision.socket && this.vision.socket.readyState <= WebSocket.OPEN) return;
      this.connectVision();
    },

    connectVision() {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/vision`);
      this.vision.socket = socket;
      this.vision.status = 'connecting';
      this.vision.lastError = '';

      socket.addEventListener('open', () => {
        this.vision.status = 'connected';
      });

      socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ack') {
          this.vision.pending = false;
          this.vision.acked += 1;
          this.vision.lastError = '';
          return;
        }
        if (message.type === 'error') {
          this.vision.pending = false;
          this.vision.lastError = message.error;
        }
      });

      socket.addEventListener('close', () => {
        this.vision.status = 'disconnected';
        this.vision.pending = false;
        if (this.running) {
          window.setTimeout(() => this.connectVision(), 1000);
        }
      });

      socket.addEventListener('error', () => {
        this.vision.status = 'error';
        this.vision.lastError = 'socket error';
      });
    },

    startLocation() {
      if (!('geolocation' in navigator)) {
        this.location.status = 'error';
        this.location.lastError = 'geolocation unavailable';
        return;
      }
      this.syncLocation();
      if (this.location.watchId !== null) return;

      this.location.status = 'connecting';
      this.location.lastError = '';
      this.location.watchId = navigator.geolocation.watchPosition(
        (position) => this.sendLocationFix(position),
        (error) => {
          this.location.status = 'error';
          this.location.pending = false;
          this.location.lastError = error.message || 'geolocation permission failed';
        },
        {
          enableHighAccuracy: true,
          maximumAge: 5000,
          timeout: 10000,
        },
      );
    },

    stopLocation() {
      if (this.location.watchId !== null && 'geolocation' in navigator) {
        navigator.geolocation.clearWatch(this.location.watchId);
      }
      this.location.watchId = null;
      this.location.pendingPosition = null;
      this.location.pending = false;
      this.disconnectLocation();
    },

    syncLocation() {
      if (!this.running) {
        this.disconnectLocation();
        return;
      }
      if (this.location.socket && this.location.socket.readyState <= WebSocket.OPEN) return;
      this.connectLocation();
    },

    connectLocation() {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/location`);
      this.location.socket = socket;
      this.location.status = 'connecting';
      this.location.lastError = '';

      socket.addEventListener('open', () => {
        this.location.status = 'connected';
        if (this.location.pendingPosition) {
          const pendingPosition = this.location.pendingPosition;
          this.location.pendingPosition = null;
          this.sendLocationFix(pendingPosition);
        }
      });

      socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ack') {
          this.location.pending = false;
          this.location.acked += 1;
          this.location.lastError = '';
          return;
        }
        if (message.type === 'error') {
          this.location.pending = false;
          this.location.lastError = message.error;
        }
      });

      socket.addEventListener('close', () => {
        this.location.status = 'disconnected';
        this.location.pending = false;
        this.location.socket = null;
        if (this.running && this.location.watchId !== null) {
          window.setTimeout(() => this.connectLocation(), 1000);
        }
      });

      socket.addEventListener('error', () => {
        this.location.status = 'error';
        this.location.lastError = 'socket error';
      });
    },

    disconnectLocation() {
      this.location.pending = false;
      if (this.location.socket) {
        this.location.socket.close();
        this.location.socket = null;
      }
      this.location.status = 'disconnected';
    },

    disconnectVision() {
      this.vision.pending = false;
      if (this.vision.socket) {
        this.vision.socket.close();
        this.vision.socket = null;
      }
      this.vision.status = 'disconnected';
    },

    async startAsr() {
      this.syncAsr();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        this.asr.status = 'error';
        this.asr.lastError = 'Web Audio unavailable';
        return;
      }
      if (!this.stream || !this.stream.getAudioTracks().length) {
        this.asr.status = 'error';
        this.asr.lastError = 'microphone track unavailable';
        return;
      }
      this.audioContext = new AudioContextClass();
      await this.audioContext.resume();
      this.asr.sampleRate = this.audioContext.sampleRate;
      this.audioSource = this.audioContext.createMediaStreamSource(this.stream);
      this.audioProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.audioSink = this.audioContext.createGain();
      this.audioSink.gain.value = 0;
      this.audioProcessor.onaudioprocess = (event) => {
        if (!this.running) return;
        const input = event.inputBuffer.getChannelData(0);
        this.collectAsrSamples(input);
      };
      this.audioSource.connect(this.audioProcessor);
      this.audioProcessor.connect(this.audioSink);
      this.audioSink.connect(this.audioContext.destination);
    },

    stopAsr() {
      if (this.audioProcessor) {
        this.audioProcessor.disconnect();
        this.audioProcessor.onaudioprocess = null;
        this.audioProcessor = null;
      }
      if (this.audioSource) {
        this.audioSource.disconnect();
        this.audioSource = null;
      }
      if (this.audioSink) {
        this.audioSink.disconnect();
        this.audioSink = null;
      }
      if (this.audioContext) {
        this.audioContext.close();
        this.audioContext = null;
      }
      this.asr.chunks = [];
      this.asr.queuedSamples = 0;
      this.disconnectAsr();
    },

    syncAsr() {
      if (!this.running) {
        this.disconnectAsr();
        return;
      }
      if (this.asr.socket && this.asr.socket.readyState <= WebSocket.OPEN) return;
      this.connectAsr();
    },

    connectAsr() {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/asr`);
      this.asr.socket = socket;
      this.asr.status = 'connecting';
      this.asr.lastError = '';

      socket.addEventListener('open', () => {
        this.asr.status = 'connected';
      });

      socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ack') {
          this.asr.acked += 1;
          this.asr.lastError = '';
          return;
        }
        if (message.type === 'error') {
          this.asr.lastError = message.error;
        }
      });

      socket.addEventListener('close', () => {
        this.asr.status = 'disconnected';
        this.asr.socket = null;
        if (this.running) {
          window.setTimeout(() => this.connectAsr(), 1000);
        }
      });

      socket.addEventListener('error', () => {
        this.asr.status = 'error';
        this.asr.lastError = 'socket error';
      });
    },

    disconnectAsr() {
      if (this.asr.socket) {
        this.asr.socket.close();
        this.asr.socket = null;
      }
      this.asr.status = 'disconnected';
    },

    collectAsrSamples(input) {
      const copy = new Float32Array(input.length);
      copy.set(input);
      this.asr.chunks.push(copy);
      this.asr.queuedSamples += copy.length;

      const clipSamples = Math.round((this.asr.sampleRate * this.asr.clipMs) / 1000);
      while (this.asr.queuedSamples >= clipSamples) {
        this.sendAsrClip(this.takeAsrSamples(clipSamples));
      }
    },

    takeAsrSamples(count) {
      const output = new Float32Array(count);
      let offset = 0;
      while (offset < count && this.asr.chunks.length) {
        const chunk = this.asr.chunks.shift();
        const needed = count - offset;
        if (chunk.length <= needed) {
          output.set(chunk, offset);
          offset += chunk.length;
          this.asr.queuedSamples -= chunk.length;
        } else {
          output.set(chunk.subarray(0, needed), offset);
          this.asr.chunks.unshift(chunk.subarray(needed));
          this.asr.queuedSamples -= needed;
          offset += needed;
        }
      }
      return output;
    },

    sendAsrClip(samples) {
      this.syncAsr();
      if (!this.asr.socket || this.asr.socket.readyState !== WebSocket.OPEN) {
        this.asr.dropped += 1;
        return;
      }

      const sequence = ++this.asrSequence;
      const clip = {
        kind: 'audio.clip',
        client_id: this.clientId,
        sensor_id: 'microphone.default',
        faculty: 'asr',
        sequence,
        occurred_at: new Date().toISOString(),
        duration_ms: this.asr.clipMs,
        sample_rate_hz: this.asr.sampleRate,
        channels: 1,
        sample_format: 'f32le',
        data: this.float32ToBase64(samples),
      };
      this.asr.socket.send(JSON.stringify(clip));
      this.asr.sent += 1;
    },

    float32ToBase64(samples) {
      const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
      let binary = '';
      const chunkSize = 0x8000;
      for (let offset = 0; offset < bytes.length; offset += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
      }
      return window.btoa(binary);
    },

    async captureFrame() {
      const video = this.$refs.video;
      if (!video.videoWidth || !video.videoHeight) return;

      const width = Math.min(this.targetWidth, video.videoWidth);
      const height = Math.round(video.videoHeight * (width / video.videoWidth));
      const canvas = this.$refs.canvas;
      canvas.width = width;
      canvas.height = height;
      canvas.getContext('2d').drawImage(video, 0, 0, width, height);

      const data = canvas.toDataURL(this.mime, this.quality);
      const sequence = ++this.sequence;
      const occurredAt = new Date().toISOString();

      if (!this.vision.socket || this.vision.socket.readyState !== WebSocket.OPEN) {
        this.vision.dropped += 1;
        return;
      }
      if (this.vision.pending) {
        this.vision.dropped += 1;
        return;
      }

      const frame = {
        kind: 'vision.frame',
        client_id: this.clientId,
        sensor_id: this.sensorId,
        faculty: 'vision',
        sequence,
        occurred_at: occurredAt,
        mime: this.mime,
        width,
        height,
        data,
      };

      this.vision.socket.send(JSON.stringify(frame));
      this.vision.pending = true;
      this.vision.sent += 1;
    },

    sendLocationFix(position) {
      this.syncLocation();
      const coords = position.coords;
      this.location.lastFix = this.formatCoordinates(coords.latitude, coords.longitude);

      if (!this.location.socket || this.location.socket.readyState !== WebSocket.OPEN) {
        this.location.pendingPosition = position;
        return;
      }
      if (this.location.pending) {
        this.location.dropped += 1;
        return;
      }

      const sequence = ++this.locationSequence;
      const occurredAt = new Date(position.timestamp || Date.now()).toISOString();
      const fix = {
        kind: 'location.fix',
        client_id: this.clientId,
        sensor_id: this.locationSensorId,
        faculty: 'location',
        sequence,
        occurred_at: occurredAt,
        latitude: coords.latitude,
        longitude: coords.longitude,
        accuracy_meters: coords.accuracy,
        altitude_meters: coords.altitude,
        altitude_accuracy_meters: coords.altitudeAccuracy,
        heading_degrees: coords.heading,
        speed_meters_per_second: coords.speed,
      };

      this.location.socket.send(JSON.stringify(fix));
      this.location.pending = true;
      this.location.sent += 1;
    },

    formatCoordinates(latitude, longitude) {
      if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return '-';
      return `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`;
    },
  };
};
