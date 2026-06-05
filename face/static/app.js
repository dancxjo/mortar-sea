window.faceApp = function faceApp() {
  return {
    cameraMessage: 'Camera idle',
    clientId: 'face-browser',
    sensorId: 'camera.default',
    faculties: [],
    fps: 3,
    experiencePrompt: '',
    experienceResponse: '',
    experienceSocket: null,
    experienceStatus: 'disconnected',
    activeExperienceGenerationId: null,
    activeVoiceGenerationId: null,
    voiceResponse: '',
    voiceStatus: 'waiting',
    llmJobs: [],
    mime: 'image/jpeg',
    quality: 0.45,
    running: false,
    sequence: 0,
    stream: null,
    targetWidth: 160,
    timer: null,

    async init() {
      const response = await fetch('/api/faculties');
      const names = await response.json();
      this.faculties = names.map((name) => ({
        name,
        enabled: true,
        socket: null,
        status: 'disconnected',
        pending: false,
        sent: 0,
        acked: 0,
        dropped: 0,
        lastError: '',
      }));
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
        this.faculties.forEach((faculty) => this.syncFaculty(faculty));
        this.scheduleCapture();
      } catch (error) {
        this.cameraMessage = error.message || 'Camera permission failed';
      }
    },

    stop() {
      this.running = false;
      window.clearTimeout(this.timer);
      this.faculties.forEach((faculty) => this.disconnect(faculty));
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
          this.activeVoiceGenerationId = message.generation_id;
          this.voiceResponse = '';
          this.voiceStatus = 'thinking';
          return;
        }
        if (message.type === 'voice_response_token') {
          if (message.generation_id !== this.activeVoiceGenerationId) return;
          this.voiceResponse += message.text;
          return;
        }
        if (message.type === 'voice_response_done') {
          if (message.generation_id !== this.activeVoiceGenerationId) return;
          this.voiceStatus = 'waiting';
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
        next.promptChars = message.prompt_chars;
        next.maxTokens = message.max_tokens;
      } else if (message.type === 'llm_job_started') {
        next.queueWaitMs = message.queue_wait_ms;
      } else if (message.type === 'llm_job_completed') {
        next.responseChars = message.response_chars;
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
    },

    visibleLlmJobs() {
      return this.llmJobs.slice(0, 8);
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
      if (milliseconds < 1000) return `${milliseconds}ms`;
      if (milliseconds < 10000) return `${(milliseconds / 1000).toFixed(1)}s`;
      return `${Math.round(milliseconds / 1000)}s`;
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

    syncFaculty(faculty) {
      if (!this.running || !faculty.enabled) {
        this.disconnect(faculty);
        return;
      }
      if (faculty.socket && faculty.socket.readyState <= WebSocket.OPEN) return;
      this.connect(faculty);
    },

    connect(faculty) {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/faculties/${faculty.name}`);
      faculty.socket = socket;
      faculty.status = 'connecting';
      faculty.lastError = '';

      socket.addEventListener('open', () => {
        faculty.status = 'connected';
      });

      socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ack') {
          faculty.pending = false;
          faculty.acked += 1;
          faculty.lastError = '';
          return;
        }
        if (message.type === 'error') {
          faculty.pending = false;
          faculty.lastError = message.error;
        }
      });

      socket.addEventListener('close', () => {
        faculty.status = 'disconnected';
        faculty.pending = false;
        if (this.running && faculty.enabled) {
          window.setTimeout(() => this.connect(faculty), 1000);
        }
      });

      socket.addEventListener('error', () => {
        faculty.status = 'error';
        faculty.lastError = 'socket error';
      });
    },

    disconnect(faculty) {
      faculty.pending = false;
      if (faculty.socket) {
        faculty.socket.close();
        faculty.socket = null;
      }
      faculty.status = 'disconnected';
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

      this.faculties.forEach((faculty) => {
        if (!faculty.enabled) return;
        if (!faculty.socket || faculty.socket.readyState !== WebSocket.OPEN) {
          faculty.dropped += 1;
          return;
        }
        if (faculty.pending) {
          faculty.dropped += 1;
          return;
        }

        const frame = {
          kind: 'vision.frame',
          client_id: this.clientId,
          sensor_id: this.sensorId,
          faculty: faculty.name,
          sequence,
          occurred_at: occurredAt,
          mime: this.mime,
          width,
          height,
          data,
        };

        faculty.socket.send(JSON.stringify(frame));
        faculty.pending = true;
        faculty.sent += 1;
      });
    },
  };
};
