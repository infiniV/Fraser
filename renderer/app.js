class FraserApp {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
    this.outputPath = '';
    this.settings = {
      model: 'yolov8m-face',
      mode: 'blur',
      confidence: 0.3
    };
  }

  init() {
    this.bindEvents();
    this.detectGPU();
    this.setStatus('Ready');
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
    });

    document.getElementById('select-mode').addEventListener('change', (e) => {
      this.settings.mode = e.target.value;
    });

    document.getElementById('select-confidence').addEventListener('change', (e) => {
      this.settings.confidence = parseFloat(e.target.value);
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

    // IPC listeners
    window.electronAPI.onProcessingProgress((data) => {
      this.updateQueueItem(data.id, { progress: data.progress, status: 'processing' });
    });

    window.electronAPI.onProcessingComplete((data) => {
      this.updateQueueItem(data.id, { progress: 100, status: 'completed' });
    });

    window.electronAPI.onProcessingError((data) => {
      this.updateQueueItem(data.id, { status: 'error' });
      this.setStatus(`Error: ${data.error}`);
    });
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
  }

  renderQueue() {
    const queueList = document.getElementById('queue-list');

    if (this.queue.length === 0) {
      queueList.innerHTML = '<p class="queue__empty">No files added. Click "Add Files" to begin.</p>';
      return;
    }

    queueList.innerHTML = this.queue.map(item => `
      <div class="queue-item" data-id="${item.id}">
        <span class="queue-item__icon">${this.getStatusIcon(item.type)}</span>
        <span class="queue-item__name" title="${item.name}">${item.name}</span>
        <span class="queue-item__duration">${item.duration}</span>
        <div class="queue-item__progress">
          <div class="queue-item__progress-fill" style="width: ${item.progress}%"></div>
        </div>
        <span class="queue-item__status queue-item__status--${item.status}">${item.status}</span>
        <button class="queue-item__remove" data-id="${item.id}">×</button>
      </div>
    `).join('');

    // Bind remove buttons
    document.querySelectorAll('.queue-item__remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = parseFloat(e.target.dataset.id);
        this.removeFromQueue(id);
      });
    });
  }

  getStatusIcon(type) {
    const icons = {
      video: '🎬',
      folder: '📁',
      rtsp: '📡'
    };
    return icons[type] || '📄';
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
      const gpuInfo = await window.electronAPI.getGPUInfo();
      const gpuText = gpuInfo.cuda ? `CUDA: ${gpuInfo.name}` : 'CPU Mode';
      document.getElementById('status-gpu').textContent = gpuText;
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
