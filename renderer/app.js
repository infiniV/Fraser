class FraserApp {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
    this.pythonReady = false;
    this.outputPath = '';
    this.settings = {
      model: 'yolov11n-face',
      mode: 'black',
      confidence: 0.25,
      detectionResolution: '360p',
      redactionColor: '#000000',
      temporalBuffer: 5
    };
    this.setupPackages = [];
    this.totalPackages = 0;
    this.installedPackages = 0;
    this.progressSocket = null;
  }

  init() {
    this.bindEvents();
    this.setStatus('Initializing...');
    document.getElementById('status-gpu').textContent = 'Starting...';
    this.showSetupOverlay();
    this.loadSavedState();
  }

  async loadSavedState() {
    try {
      const state = await window.electronAPI.loadQueue();
      if (state && state.queue && state.queue.length > 0) {
        this.queue = state.queue;
        this.outputPath = state.outputPath || '';
        this.settings = state.settings || this.settings;

        if (this.outputPath) {
          document.getElementById('output-path').value = this.outputPath;
        }

        // Update settings dropdowns
        if (state.settings) {
          document.getElementById('select-model').value = state.settings.model;
          document.getElementById('select-mode').value = state.settings.mode;
          document.getElementById('select-confidence').value = state.settings.confidence.toString();
          if (state.settings.detectionResolution) {
            document.getElementById('resolution-select').value = state.settings.detectionResolution;
          }
          if (state.settings.redactionColor) {
            this.settings.redactionColor = state.settings.redactionColor;
          }
          if (state.settings.temporalBuffer) {
            this.settings.temporalBuffer = state.settings.temporalBuffer;
          }
        }

        this.renderQueue();
        console.log('Restored queue state:', this.queue.length, 'items');
      }
    } catch (error) {
      console.error('Error loading saved state:', error);
    }
  }

  async saveState() {
    try {
      await window.electronAPI.saveQueue({
        queue: this.queue,
        outputPath: this.outputPath,
        settings: this.settings
      });
    } catch (error) {
      console.error('Error saving state:', error);
    }
  }

  showSetupOverlay() {
    document.getElementById('setup-overlay').classList.remove('hidden');
  }

  hideSetupOverlay() {
    document.getElementById('setup-overlay').classList.add('hidden');
  }

  updateSetupProgress(message) {
    const messageEl = document.getElementById('setup-message');
    const packagesEl = document.getElementById('setup-packages');
    const progressFill = document.getElementById('setup-progress-fill');
    const progressText = document.getElementById('setup-progress-text');

    messageEl.textContent = message;

    // Parse uv output for package info
    const resolvedMatch = message.match(/Resolved (\d+) packages/);
    if (resolvedMatch) {
      this.totalPackages = parseInt(resolvedMatch[1]);
    }

    const installedMatch = message.match(/Installed (\d+) packages?/);
    if (installedMatch) {
      this.installedPackages = parseInt(installedMatch[1]);
    }

    // Check for download progress
    const downloadMatch = message.match(/Downloading|Prepared|Installing/i);
    if (downloadMatch || resolvedMatch) {
      // Show indeterminate progress while downloading
      if (this.totalPackages > 0 && this.installedPackages > 0) {
        const percent = Math.round((this.installedPackages / this.totalPackages) * 100);
        progressFill.style.width = `${percent}%`;
        progressText.textContent = `${percent}%`;
      } else if (resolvedMatch) {
        progressFill.style.width = '20%';
        progressText.textContent = 'Downloading...';
      }
    }

    // Track package names
    const packageMatch = message.match(/([a-z0-9_-]+)==[\d.]+/gi);
    if (packageMatch) {
      packageMatch.forEach(pkg => {
        if (!this.setupPackages.includes(pkg)) {
          this.setupPackages.push(pkg);
          const item = document.createElement('div');
          item.className = 'setup-packages__item';
          item.textContent = pkg;
          packagesEl.appendChild(item);
          packagesEl.scrollTop = packagesEl.scrollHeight;
        }
      });
    }

    // Check for specific stages
    if (message.includes('Creating virtual environment')) {
      progressFill.style.width = '10%';
      progressText.textContent = '10%';
    } else if (message.includes('Installing requirements')) {
      progressFill.style.width = '15%';
      progressText.textContent = '15%';
    } else if (message.includes('Starting Python server')) {
      progressFill.style.width = '90%';
      progressText.textContent = '90%';
    } else if (message.includes('setup complete')) {
      progressFill.style.width = '100%';
      progressText.textContent = '100%';
    }
  }

  bindEvents() {
    // Window controls
    document.getElementById('btn-minimize').addEventListener('click', () => {
      window.electronAPI.minimize();
    });

    document.getElementById('btn-maximize').addEventListener('click', () => {
      window.electronAPI.maximize();
    });

    document.getElementById('btn-close').addEventListener('click', () => {
      window.electronAPI.close();
    });

    // Add buttons
    document.getElementById('btn-add-files').addEventListener('click', () => {
      this.addFiles();
    });

    document.getElementById('btn-add-folder').addEventListener('click', () => {
      this.addFolder();
    });

    document.getElementById('btn-add-rtsp').addEventListener('click', () => {
      const url = prompt('Enter RTSP stream URL:');
      if (url) {
        this.addToQueue({ path: url, type: 'rtsp', name: url, duration: 'Live' });
      }
    });

    // Settings
    document.getElementById('select-model').addEventListener('change', (e) => {
      this.settings.model = e.target.value;
      this.saveState();
    });

    document.getElementById('select-mode').addEventListener('change', (e) => {
      this.settings.mode = e.target.value;
      this.saveState();
    });

    document.getElementById('select-confidence').addEventListener('change', (e) => {
      this.settings.confidence = parseFloat(e.target.value);
      this.saveState();
    });

    document.getElementById('resolution-select').addEventListener('change', (e) => {
      this.settings.detectionResolution = e.target.value;
      this.saveState();
    });

    // Output
    document.getElementById('btn-output').addEventListener('click', () => {
      this.selectOutput();
    });

    // Actions
    document.getElementById('btn-start').addEventListener('click', () => {
      this.startProcessing();
    });

    document.getElementById('btn-pause').addEventListener('click', () => {
      this.pauseProcessing();
    });

    // Setup progress listener
    window.electronAPI.onSetupProgress((_, message) => {
      this.setStatus(message);
      this.updateSetupProgress(message);
    });

    // Python ready listener
    window.electronAPI.onPythonReady(() => {
      this.setStatus('Ready');
      this.pythonReady = true;
      this.hideSetupOverlay();
      this.detectGPU();
    });

    // Python error listener
    window.electronAPI.onPythonError((_, error) => {
      this.setStatus(`Error: ${error}`);
      document.getElementById('status-gpu').textContent = 'Server Error';
      document.getElementById('setup-message').textContent = `Error: ${error}`;
      document.getElementById('setup-progress-fill').style.background = 'var(--destructive)';
    });

    // Processing progress listener
    window.electronAPI.onProcessProgress((_, data) => {
      this.handleProcessProgress(data);
    });
  }

  handleProcessProgress(data) {
    const { itemId, progress, status, jobId, stats, error, fps, facesInFrame } = data;

    // Find the queue item
    const item = this.queue.find(q => q.id === itemId);
    if (!item) return;

    if (status === 'processing') {
      item.status = 'processing';
      item.progress = progress || 0;
      item.jobId = jobId;
      if (fps) {
        this.setStatus(`Processing ${item.name} - ${fps} FPS`);
      }
    } else if (status === 'completed') {
      item.status = 'completed';
      item.progress = 100;
      if (stats) {
        item.faces = stats.faces_detected;
        item.duration = `${Math.round(stats.processing_time)}s`;
      }
      this.setStatus(`Completed: ${item.name}`);
      this.saveState();
      this.checkAllCompleted();
    } else if (status === 'error') {
      item.status = 'error';
      item.error = error;
      this.setStatus(`Error: ${error}`);
      this.saveState();
    }

    this.renderQueue();
  }

  checkAllCompleted() {
    const allDone = this.queue.every(item =>
      item.status === 'completed' || item.status === 'error'
    );

    if (allDone && this.isProcessing) {
      this.isProcessing = false;
      document.getElementById('btn-start').disabled = false;
      document.getElementById('btn-pause').disabled = true;
      this.setStatus('All jobs completed');
    }
  }

  async addFiles() {
    try {
      const result = await window.electronAPI.selectFiles();
      if (result.canceled) return;

      for (const filePath of result.filePaths) {
        const name = filePath.split(/[\\/]/).pop();
        this.addToQueue({
          path: filePath,
          type: 'video',
          name: name,
          duration: '00:00'
        });
      }
    } catch (error) {
      console.error('Error adding files:', error);
      this.setStatus('Error adding files');
    }
  }

  async addFolder() {
    try {
      const result = await window.electronAPI.selectFolder();
      if (result.canceled) return;

      const folderPath = result.filePaths[0];
      const name = folderPath.split(/[\\/]/).pop();

      this.addToQueue({
        path: folderPath,
        type: 'folder',
        name: name,
        duration: 'Multiple'
      });
    } catch (error) {
      console.error('Error adding folder:', error);
      this.setStatus('Error adding folder');
    }
  }

  async selectOutput() {
    try {
      const result = await window.electronAPI.selectFolder();
      if (result.canceled) return;

      this.outputPath = result.filePaths[0];
      document.getElementById('output-path').value = this.outputPath;
      this.saveState();
    } catch (error) {
      console.error('Error selecting output:', error);
      this.setStatus('Error selecting output folder');
    }
  }

  addToQueue(item) {
    const id = Date.now() + Math.random();
    const queueItem = {
      id,
      ...item,
      progress: 0,
      status: 'pending'
    };

    this.queue.push(queueItem);
    this.renderQueue();
    this.saveState();
  }

  renderQueue() {
    const queueList = document.getElementById('queue-list');

    if (this.queue.length === 0) {
      queueList.innerHTML = '<p class="queue__empty">No files added. Click "Add Files" to begin.</p>';
      return;
    }

    queueList.innerHTML = this.queue.map(item => `
      <div class="queue-item queue-item--${item.status}" data-id="${item.id}">
        <span class="queue-item__icon">${this.getStatusIcon(item.status, item.type)}</span>
        <span class="queue-item__name" title="${item.path || item.name}">${item.name}</span>
        <span class="queue-item__info">${this.getItemInfo(item)}</span>
        <div class="queue-item__progress">
          <div class="queue-item__progress-fill queue-item__progress-fill--${item.status}" style="width: ${item.progress}%"></div>
        </div>
        <span class="queue-item__status queue-item__status--${item.status}">${this.getStatusText(item)}</span>
        <button class="queue-item__remove" data-id="${item.id}" ${item.status === 'processing' ? 'disabled' : ''}>x</button>
      </div>
    `).join('');

    // Bind remove buttons
    document.querySelectorAll('.queue-item__remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        if (e.target.disabled) return;
        const id = parseFloat(e.target.dataset.id);
        this.removeFromQueue(id);
      });
    });
  }

  getStatusIcon(status, type) {
    const statusIcons = {
      completed: '[ok]',
      processing: '[>>]',
      error: '[!!]',
      pending: '[..]'
    };
    return statusIcons[status] || statusIcons.pending;
  }

  getItemInfo(item) {
    if (item.status === 'completed' && item.faces !== undefined) {
      return `${item.faces} faces`;
    }
    if (item.status === 'processing') {
      return `${Math.round(item.progress)}%`;
    }
    return item.duration || '--';
  }

  getStatusText(item) {
    if (item.status === 'completed') return 'Done';
    if (item.status === 'processing') return 'Processing';
    if (item.status === 'error') return 'Error';
    return 'Pending';
  }

  updateQueueItem(id, updates) {
    const item = this.queue.find(q => q.id === id);
    if (item) {
      Object.assign(item, updates);
      this.renderQueue();
    }
  }

  removeFromQueue(id) {
    this.queue = this.queue.filter(item => item.id !== id);
    this.renderQueue();
    this.saveState();
  }

  async startProcessing() {
    if (this.queue.length === 0) {
      alert('Please add files to the queue first');
      return;
    }

    if (!this.outputPath) {
      alert('Please select an output folder');
      return;
    }

    this.isProcessing = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-pause').disabled = false;
    this.setStatus('Processing...');

    try {
      await window.electronAPI.startProcessing({
        queue: this.queue,
        outputPath: this.outputPath,
        settings: this.settings
      });
    } catch (error) {
      console.error('Error starting processing:', error);
      this.setStatus('Error starting processing');
      this.pauseProcessing();
    }
  }

  pauseProcessing() {
    this.isProcessing = false;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-pause').disabled = true;
    this.setStatus('Paused');

    window.electronAPI.pauseProcessing();
  }

  setStatus(text) {
    document.getElementById('status-text').textContent = text;
  }

  async detectGPU() {
    try {
      const status = await window.electronAPI.getPythonStatus();
      if (status.running && status.device) {
        document.getElementById('status-gpu').textContent = status.device;
      } else {
        document.getElementById('status-gpu').textContent = 'Detecting GPU...';
      }
    } catch (error) {
      console.error('Error detecting GPU:', error);
      document.getElementById('status-gpu').textContent = 'GPU: Unknown';
    }
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new FraserApp();
  app.init();
});
