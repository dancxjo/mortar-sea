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
    llmJobs: [],
    selectedLlmJobId: null,
    mime: 'image/jpeg',
    quality: 0.45,
    running: false,
    sequence: 0,
    locationSequence: 0,
    stream: null,
    targetWidth: 160,
    timer: null,

    async init() {
      this.connectRealtimeExperience();
    },

    async start() {
      this.cameraMessage = 'Requesting camera';
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: this.targetWidth },
            facingMode: 'user',
          },
          audio: false,
        });
        this.$refs.video.srcObject = this.stream;
        await this.$refs.video.play();
        this.running = true;
        this.cameraMessage = 'Camera running';
        this.syncVision();
        this.startLocation();
        this.scheduleCapture();
      } catch (error) {
        this.cameraMessage = error.message || 'Camera permission failed';
      }
    },

    stop() {
      this.running = false;
      window.clearTimeout(this.timer);
      this.disconnectVision();
      this.stopLocation();
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      this.cameraMessage = 'Camera idle';
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
          if (message.generation_id !== this.activeVoiceGenerationId) return;
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
          this.voiceStatus = 'waiting';
          return;
        }
        if (message.type === 'voice_observation') {
          this.activeVoiceGenerationId = message.generation_id;
          if (message.observation?.emoji) {
            this.faceEmoji = message.observation.emoji;
          }
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
