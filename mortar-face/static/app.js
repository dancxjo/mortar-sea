window.faceApp = function faceApp() {
  return {
    cameraMessage: 'Camera idle',
    clientId: 'face-browser',
    sensorId: 'camera.default',
    faculties: [],
    fps: 3,
    mime: 'image/jpeg',
    quality: 0.72,
    running: false,
    sequence: 0,
    stream: null,
    targetWidth: 640,
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
