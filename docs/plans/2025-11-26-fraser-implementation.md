# Fraser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a desktop app for batch video face anonymization using Electron + Python, following ComfyUI Desktop architecture patterns.

**Architecture:** Electron shell manages Python backend lifecycle. Python (FastAPI + Ultralytics YOLO) handles video processing. WebSocket for real-time progress. Queue state persisted for crash recovery.

**Tech Stack:** Electron, TypeScript, Vite, Python 3.12, FastAPI, Ultralytics, PyAV, uv package manager

---

## Phase 1: Project Scaffolding

### Task 1: Initialize Electron + Vite Project

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `vite.config.ts`
- Create: `vite.preload.config.ts`

**Step 1: Create package.json**

```json
{
  "name": "fraser",
  "version": "0.1.0",
  "description": "Face anonymization for video files",
  "main": ".vite/build/main.cjs",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build && vite build --config vite.preload.config.ts",
    "start": "yarn build && electron .",
    "lint": "eslint src --ext .ts"
  },
  "devDependencies": {
    "@types/node": "^22.10.0",
    "electron": "^31.3.1",
    "typescript": "^5.5.0",
    "vite": "^6.0.0"
  },
  "dependencies": {
    "electron-store": "^8.2.0",
    "electron-log": "^5.2.0"
  }
}
```

**Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

**Step 3: Create vite.config.ts**

```typescript
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    outDir: '.vite/build',
    lib: {
      entry: resolve(__dirname, 'src/main.ts'),
      formats: ['cjs'],
      fileName: () => 'main.cjs',
    },
    rollupOptions: {
      external: ['electron', 'electron-store', 'electron-log', 'node:child_process', 'node:path', 'node:fs', 'node:os'],
    },
    minify: false,
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
});
```

**Step 4: Create vite.preload.config.ts**

```typescript
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    outDir: '.vite/build',
    lib: {
      entry: resolve(__dirname, 'src/preload.ts'),
      formats: ['cjs'],
      fileName: () => 'preload.cjs',
    },
    rollupOptions: {
      external: ['electron'],
    },
    minify: false,
  },
});
```

**Step 5: Install dependencies**

Run: `yarn install`
Expected: Dependencies installed successfully

**Step 6: Commit**

```bash
git add package.json tsconfig.json vite.config.ts vite.preload.config.ts
git commit -m "chore: initialize electron + vite project structure"
```

---

### Task 2: Create Electron Main Entry Point

**Files:**
- Create: `src/main.ts`
- Create: `src/preload.ts`
- Create: `src/constants.ts`

**Step 1: Create src/constants.ts**

```typescript
export const IPC_CHANNELS = {
  // Dialog
  SELECT_FILES: 'dialog:selectFiles',
  SELECT_FOLDER: 'dialog:selectFolder',
  SELECT_OUTPUT: 'dialog:selectOutputDir',

  // App
  APP_INFO: 'app:info',
  APP_QUIT: 'app:quit',

  // Python
  PYTHON_STATUS: 'python:status',
  PYTHON_READY: 'python:ready',
  PYTHON_ERROR: 'python:error',
  PYTHON_RESTART: 'python:restart',

  // Settings
  SETTINGS_GET: 'settings:get',
  SETTINGS_SAVE: 'settings:save',

  // Install
  INSTALL_PROGRESS: 'install:progress',

  // Window
  WINDOW_MINIMIZE: 'window:minimize',
  WINDOW_MAXIMIZE: 'window:maximize',
  WINDOW_CLOSE: 'window:close',
} as const;

export const PYTHON_PORT = 8420;

export const SUPPORTED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'];
```

**Step 2: Create src/preload.ts**

```typescript
import { contextBridge, ipcRenderer } from 'electron';
import { IPC_CHANNELS } from './constants';

contextBridge.exposeInMainWorld('electronAPI', {
  // File dialogs
  selectFiles: () => ipcRenderer.invoke(IPC_CHANNELS.SELECT_FILES),
  selectFolder: () => ipcRenderer.invoke(IPC_CHANNELS.SELECT_FOLDER),
  selectOutputDir: () => ipcRenderer.invoke(IPC_CHANNELS.SELECT_OUTPUT),

  // App
  getAppInfo: () => ipcRenderer.invoke(IPC_CHANNELS.APP_INFO),
  quit: () => ipcRenderer.send(IPC_CHANNELS.APP_QUIT),

  // Python
  getPythonStatus: () => ipcRenderer.invoke(IPC_CHANNELS.PYTHON_STATUS),
  restartPython: () => ipcRenderer.invoke(IPC_CHANNELS.PYTHON_RESTART),

  // Settings
  getSettings: () => ipcRenderer.invoke(IPC_CHANNELS.SETTINGS_GET),
  saveSettings: (settings: Record<string, unknown>) =>
    ipcRenderer.invoke(IPC_CHANNELS.SETTINGS_SAVE, settings),

  // Window
  minimize: () => ipcRenderer.send(IPC_CHANNELS.WINDOW_MINIMIZE),
  maximize: () => ipcRenderer.send(IPC_CHANNELS.WINDOW_MAXIMIZE),
  close: () => ipcRenderer.send(IPC_CHANNELS.WINDOW_CLOSE),

  // Events
  onPythonReady: (callback: () => void) =>
    ipcRenderer.on(IPC_CHANNELS.PYTHON_READY, callback),
  onPythonError: (callback: (_: unknown, error: string) => void) =>
    ipcRenderer.on(IPC_CHANNELS.PYTHON_ERROR, callback),
  onInstallProgress: (callback: (_: unknown, progress: number) => void) =>
    ipcRenderer.on(IPC_CHANNELS.INSTALL_PROGRESS, callback),
});
```

**Step 3: Create src/main.ts**

```typescript
import { app, BrowserWindow } from 'electron';
import path from 'node:path';
import log from 'electron-log/main';

log.initialize();
log.info(`Starting Fraser v${app.getVersion()}`);

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    minWidth: 800,
    minHeight: 600,
    backgroundColor: '#0a0a0a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    frame: process.platform === 'darwin',
  });

  // Load renderer
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
```

**Step 4: Test build**

Run: `yarn build`
Expected: Build completes with files in `.vite/build/`

**Step 5: Commit**

```bash
git add src/main.ts src/preload.ts src/constants.ts
git commit -m "feat: add electron main entry point with preload bridge"
```

---

### Task 3: Create Renderer (Frontend) Structure

**Files:**
- Create: `renderer/index.html`
- Create: `renderer/styles/theme.css`
- Create: `renderer/styles/main.css`
- Create: `renderer/app.js`

**Step 1: Create renderer/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';">
  <title>Fraser</title>
  <link rel="stylesheet" href="styles/theme.css">
  <link rel="stylesheet" href="styles/main.css">
</head>
<body>
  <div id="app">
    <header class="header">
      <h1 class="header__title">Fraser</h1>
      <div class="header__controls">
        <button id="btn-minimize" class="header__btn">-</button>
        <button id="btn-maximize" class="header__btn">=</button>
        <button id="btn-close" class="header__btn">x</button>
      </div>
    </header>

    <main class="main">
      <section class="add-section">
        <button id="btn-add-files" class="btn btn--primary">+ Add Files</button>
        <button id="btn-add-folder" class="btn btn--primary">+ Add Folder</button>
        <button id="btn-add-rtsp" class="btn btn--secondary">+ RTSP Stream</button>
      </section>

      <section class="settings-section">
        <div class="setting">
          <label class="setting__label">Model</label>
          <select id="select-model" class="setting__select">
            <option value="yolov8n-face">YOLOv8 Nano (Fast)</option>
            <option value="yolov8m-face" selected>YOLOv8 Medium</option>
            <option value="yolov8l-face">YOLOv8 Large (Accurate)</option>
            <option value="yolo11n-face">YOLO11 Nano</option>
          </select>
        </div>
        <div class="setting">
          <label class="setting__label">Mode</label>
          <select id="select-mode" class="setting__select">
            <option value="blur" selected>Blur</option>
            <option value="black">Black Rectangle</option>
            <option value="color">Solid Color</option>
          </select>
        </div>
        <div class="setting">
          <label class="setting__label">Confidence</label>
          <select id="select-confidence" class="setting__select">
            <option value="0.2">0.2 (Catch all)</option>
            <option value="0.3" selected>0.3 (Recommended)</option>
            <option value="0.5">0.5 (Balanced)</option>
            <option value="0.7">0.7 (High precision)</option>
          </select>
        </div>
      </section>

      <section class="queue-section">
        <h2 class="queue__title">Queue</h2>
        <div id="queue-list" class="queue__list">
          <p class="queue__empty">No files added. Click "Add Files" to begin.</p>
        </div>
      </section>

      <section class="output-section">
        <label class="output__label">Output Folder:</label>
        <input type="text" id="output-path" class="output__input" readonly placeholder="Select output folder...">
        <button id="btn-output" class="btn btn--secondary">Browse</button>
      </section>

      <section class="actions-section">
        <button id="btn-start" class="btn btn--primary btn--large">Start</button>
        <button id="btn-pause" class="btn btn--secondary btn--large" disabled>Pause</button>
      </section>
    </main>

    <footer class="status-bar">
      <span id="status-text">Ready</span>
      <span id="status-gpu">Detecting GPU...</span>
    </footer>
  </div>

  <script src="app.js" type="module"></script>
</body>
</html>
```

**Step 2: Create renderer/styles/theme.css**

```css
:root {
  --background: #0a0a0a;
  --foreground: #f5f5f5;
  --card: #121212;
  --card-foreground: #f5f5f5;
  --primary: #7c9082;
  --primary-foreground: #000000;
  --secondary: #1a1a1a;
  --secondary-foreground: #f5f5f5;
  --muted: #1a1a1a;
  --muted-foreground: #a0a0a0;
  --accent: #36443a;
  --accent-foreground: #f5f5f5;
  --destructive: #ef4444;
  --destructive-foreground: #ffffff;
  --border: #2a2a2a;
  --input: #121212;
  --ring: #7c9082;
  --radius: 0.35rem;
  --font-sans: 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Consolas', monospace;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: var(--font-sans);
  background: var(--background);
  color: var(--foreground);
  line-height: 1.5;
  overflow: hidden;
}
```

**Step 3: Create renderer/styles/main.css**

```css
#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: var(--card);
  border-bottom: 1px solid var(--border);
  -webkit-app-region: drag;
}

.header__title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--primary);
}

.header__controls {
  display: flex;
  gap: 0.5rem;
  -webkit-app-region: no-drag;
}

.header__btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: var(--radius);
  background: var(--secondary);
  color: var(--foreground);
  cursor: pointer;
  font-size: 0.875rem;
}

.header__btn:hover {
  background: var(--accent);
}

.main {
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.btn {
  padding: 0.5rem 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: 0.875rem;
  cursor: pointer;
  transition: background 0.15s;
}

.btn--primary {
  background: var(--primary);
  color: var(--primary-foreground);
  border-color: var(--primary);
}

.btn--primary:hover {
  background: #6a7d70;
}

.btn--secondary {
  background: var(--secondary);
  color: var(--secondary-foreground);
}

.btn--secondary:hover {
  background: var(--accent);
}

.btn--large {
  padding: 0.75rem 2rem;
  font-size: 1rem;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.add-section {
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  background: var(--card);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}

.settings-section {
  display: flex;
  gap: 1rem;
}

.setting {
  flex: 1;
}

.setting__label {
  display: block;
  font-size: 0.75rem;
  color: var(--muted-foreground);
  margin-bottom: 0.25rem;
}

.setting__select {
  width: 100%;
  padding: 0.5rem;
  background: var(--input);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--foreground);
  font-size: 0.875rem;
}

.queue-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 200px;
}

.queue__title {
  font-size: 0.875rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.queue__list {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.5rem;
  overflow-y: auto;
}

.queue__empty {
  color: var(--muted-foreground);
  text-align: center;
  padding: 2rem;
  font-size: 0.875rem;
}

.queue-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  border-radius: var(--radius);
}

.queue-item:hover {
  background: var(--muted);
}

.queue-item__icon {
  font-size: 0.875rem;
  width: 1rem;
  text-align: center;
}

.queue-item__icon--done { color: var(--primary); }
.queue-item__icon--processing { color: var(--primary); }
.queue-item__icon--pending { color: var(--muted-foreground); }
.queue-item__icon--error { color: var(--destructive); }

.queue-item__name {
  flex: 1;
  font-size: 0.875rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.queue-item__duration {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--muted-foreground);
}

.queue-item__progress {
  width: 100px;
  height: 4px;
  background: var(--muted);
  border-radius: 2px;
  overflow: hidden;
}

.queue-item__progress-fill {
  height: 100%;
  background: var(--primary);
  transition: width 0.3s;
}

.queue-item__status {
  font-size: 0.75rem;
  color: var(--muted-foreground);
  min-width: 60px;
  text-align: right;
}

.queue-item__remove {
  background: none;
  border: none;
  color: var(--muted-foreground);
  cursor: pointer;
  font-size: 1rem;
  padding: 0.25rem;
}

.queue-item__remove:hover {
  color: var(--destructive);
}

.output-section {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.output__label {
  font-size: 0.875rem;
  color: var(--muted-foreground);
}

.output__input {
  flex: 1;
  padding: 0.5rem;
  background: var(--input);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--foreground);
  font-size: 0.875rem;
}

.actions-section {
  display: flex;
  gap: 0.5rem;
}

.status-bar {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  background: var(--card);
  border-top: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--muted-foreground);
}
```

**Step 4: Create renderer/app.js**

```javascript
// Fraser Renderer Entry Point

class FraserApp {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
    this.outputPath = '';
    this.init();
  }

  init() {
    this.bindEvents();
    this.detectGPU();
  }

  bindEvents() {
    // Window controls
    document.getElementById('btn-minimize').onclick = () => window.electronAPI?.minimize();
    document.getElementById('btn-maximize').onclick = () => window.electronAPI?.maximize();
    document.getElementById('btn-close').onclick = () => window.electronAPI?.close();

    // Add files/folders
    document.getElementById('btn-add-files').onclick = () => this.addFiles();
    document.getElementById('btn-add-folder').onclick = () => this.addFolder();
    document.getElementById('btn-output').onclick = () => this.selectOutput();

    // Actions
    document.getElementById('btn-start').onclick = () => this.startProcessing();
    document.getElementById('btn-pause').onclick = () => this.pauseProcessing();
  }

  async addFiles() {
    const files = await window.electronAPI?.selectFiles();
    if (files?.length) {
      files.forEach(f => this.addToQueue(f));
    }
  }

  async addFolder() {
    const folder = await window.electronAPI?.selectFolder();
    if (folder) {
      this.setStatus(`Added folder: ${folder}`);
    }
  }

  async selectOutput() {
    const path = await window.electronAPI?.selectOutputDir();
    if (path) {
      this.outputPath = path;
      document.getElementById('output-path').value = path;
    }
  }

  addToQueue(filePath) {
    const id = Date.now().toString();
    const name = filePath.split(/[/\\]/).pop();
    this.queue.push({ id, path: filePath, name, status: 'pending' });
    this.renderQueue();
  }

  renderQueue() {
    const list = document.getElementById('queue-list');

    if (this.queue.length === 0) {
      list.innerHTML = '<p class="queue__empty">No files added. Click "Add Files" to begin.</p>';
      return;
    }

    list.innerHTML = this.queue.map(item => `
      <div class="queue-item" data-id="${item.id}">
        <span class="queue-item__icon queue-item__icon--${item.status}">${this.getStatusIcon(item.status)}</span>
        <span class="queue-item__name" title="${item.path}">${item.name}</span>
        <span class="queue-item__duration">${item.duration || '--:--'}</span>
        ${item.status === 'processing'
          ? `<div class="queue-item__progress"><div class="queue-item__progress-fill" style="width:${item.progress || 0}%"></div></div>`
          : `<span class="queue-item__status">${item.faces ?? '--'} faces</span>`
        }
        <button class="queue-item__remove" onclick="app.removeFromQueue('${item.id}')">x</button>
      </div>
    `).join('');
  }

  getStatusIcon(status) {
    const icons = { done: '✓', processing: '▶', pending: '○', error: '✗' };
    return icons[status] || '○';
  }

  removeFromQueue(id) {
    this.queue = this.queue.filter(item => item.id !== id);
    this.renderQueue();
  }

  startProcessing() {
    if (!this.outputPath) {
      alert('Please select an output folder first.');
      return;
    }
    this.isProcessing = true;
    this.setStatus('Processing...');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-pause').disabled = false;
  }

  pauseProcessing() {
    this.isProcessing = false;
    this.setStatus('Paused');
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-pause').disabled = true;
  }

  setStatus(text) {
    document.getElementById('status-text').textContent = text;
  }

  async detectGPU() {
    const gpuEl = document.getElementById('status-gpu');
    gpuEl.textContent = 'GPU: Detecting...';

    // Will be populated when Python backend connects
    setTimeout(() => {
      gpuEl.textContent = 'GPU: Waiting for backend...';
    }, 2000);
  }
}

const app = new FraserApp();
window.app = app;
```

**Step 5: Test renderer loads**

Run: `yarn start`
Expected: Electron window opens with Fraser UI

**Step 6: Commit**

```bash
git add renderer/
git commit -m "feat: add renderer with theme and basic UI components"
```

---

### Task 4: Add File Dialog Handlers

**Files:**
- Create: `src/handlers/fileHandlers.ts`
- Modify: `src/main.ts`

**Step 1: Create src/handlers/fileHandlers.ts**

```typescript
import { ipcMain, dialog, BrowserWindow } from 'electron';
import { IPC_CHANNELS, SUPPORTED_VIDEO_EXTENSIONS } from '../constants';

export function registerFileHandlers(mainWindow: BrowserWindow): void {
  ipcMain.handle(IPC_CHANNELS.SELECT_FILES, async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Select Video Files',
      properties: ['openFile', 'multiSelections'],
      filters: [
        {
          name: 'Video Files',
          extensions: SUPPORTED_VIDEO_EXTENSIONS
        },
        { name: 'All Files', extensions: ['*'] }
      ],
    });
    return result.canceled ? [] : result.filePaths;
  });

  ipcMain.handle(IPC_CHANNELS.SELECT_FOLDER, async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Select Folder with Videos',
      properties: ['openDirectory'],
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle(IPC_CHANNELS.SELECT_OUTPUT, async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Select Output Folder',
      properties: ['openDirectory', 'createDirectory'],
    });
    return result.canceled ? null : result.filePaths[0];
  });
}
```

**Step 2: Update src/main.ts to register handlers**

```typescript
import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import log from 'electron-log/main';
import { registerFileHandlers } from './handlers/fileHandlers';
import { IPC_CHANNELS } from './constants';

log.initialize();
log.info(`Starting Fraser v${app.getVersion()}`);

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    minWidth: 800,
    minHeight: 600,
    backgroundColor: '#0a0a0a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    frame: process.platform === 'darwin',
  });

  // Register IPC handlers
  registerFileHandlers(mainWindow);
  registerWindowHandlers(mainWindow);

  // Load renderer
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function registerWindowHandlers(window: BrowserWindow): void {
  ipcMain.on(IPC_CHANNELS.WINDOW_MINIMIZE, () => window.minimize());
  ipcMain.on(IPC_CHANNELS.WINDOW_MAXIMIZE, () => {
    if (window.isMaximized()) {
      window.unmaximize();
    } else {
      window.maximize();
    }
  });
  ipcMain.on(IPC_CHANNELS.WINDOW_CLOSE, () => window.close());
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
```

**Step 3: Test file dialogs work**

Run: `yarn start`
Expected: Clicking "Add Files" opens file picker dialog

**Step 4: Commit**

```bash
git add src/handlers/fileHandlers.ts src/main.ts
git commit -m "feat: add file dialog IPC handlers"
```

---

## Phase 2: Python Backend

### Task 5: Create Python Server Structure

**Files:**
- Create: `python/server.py`
- Create: `python/requirements.txt`

**Step 1: Create python/requirements.txt**

```
fastapi>=0.109.0
uvicorn>=0.27.0
ultralytics>=8.1.0
av>=12.0.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
torch>=2.1.0
torchvision>=0.16.0
```

**Step 2: Create python/server.py**

```python
"""Fraser Python Backend Server."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    """Request to process a video file."""
    file: str
    output_dir: str
    model: str = "yolov8m-face"
    mode: str = "blur"
    color: str = "#000000"
    confidence: float = 0.3
    padding: float = 0.1


class DeviceInfo(BaseModel):
    """GPU/CPU device information."""
    type: str
    name: str
    cuda_available: bool
    mps_available: bool


def get_device_info() -> DeviceInfo:
    """Detect available compute device."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_available:
        return DeviceInfo(
            type="cuda",
            name=torch.cuda.get_device_name(0),
            cuda_available=True,
            mps_available=False,
        )
    elif mps_available:
        return DeviceInfo(
            type="mps",
            name="Apple Silicon",
            cuda_available=False,
            mps_available=True,
        )
    else:
        return DeviceInfo(
            type="cpu",
            name="CPU",
            cuda_available=False,
            mps_available=False,
        )


# WebSocket connections
active_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Fraser backend starting...")
    device = get_device_info()
    print(f"Device: {device.type} ({device.name})")
    yield
    print("Fraser backend shutting down...")


app = FastAPI(title="Fraser Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    device = get_device_info()
    return {"status": "ok", "device": device}


@app.get("/models")
async def list_models():
    """List available face detection models."""
    return {
        "models": [
            {"id": "yolov8n-face", "name": "YOLOv8 Nano", "size": "6MB", "speed": "fastest"},
            {"id": "yolov8m-face", "name": "YOLOv8 Medium", "size": "50MB", "speed": "fast"},
            {"id": "yolov8l-face", "name": "YOLOv8 Large", "size": "85MB", "speed": "moderate"},
            {"id": "yolo11n-face", "name": "YOLO11 Nano", "size": "12MB", "speed": "fast"},
        ]
    }


@app.post("/process")
async def start_processing(request: ProcessRequest):
    """Start processing a video file."""
    job_id = f"job_{asyncio.get_event_loop().time():.0f}"

    # TODO: Add to processing queue

    return {"job_id": job_id, "status": "queued"}


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    # TODO: Implement cancellation
    return {"status": "cancelled", "job_id": job_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time progress updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Send initial device info
        device = get_device_info()
        await websocket.send_json({
            "type": "connected",
            "device": device.model_dump(),
        })

        while True:
            # Keep connection alive, receive commands
            data = await websocket.receive_text()
            print(f"Received: {data}")

    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_progress(message: dict):
    """Send progress update to all connected clients."""
    for connection in active_connections:
        await connection.send_json(message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8420)
```

**Step 3: Test Python server starts**

Run: `cd python && python server.py`
Expected: Server starts on http://127.0.0.1:8420

**Step 4: Test health endpoint**

Run: `curl http://127.0.0.1:8420/health`
Expected: `{"status":"ok","device":{...}}`

**Step 5: Commit**

```bash
git add python/
git commit -m "feat: add FastAPI backend server with health and models endpoints"
```

---

### Task 6: Implement Video Processing Pipeline

**Files:**
- Create: `python/processing/pipeline.py`
- Create: `python/processing/anonymizer.py`
- Create: `python/processing/__init__.py`

**Step 1: Create python/processing/__init__.py**

```python
"""Fraser video processing module."""

from .pipeline import VideoProcessor
from .anonymizer import Anonymizer

__all__ = ["VideoProcessor", "Anonymizer"]
```

**Step 2: Create python/processing/anonymizer.py**

```python
"""Face anonymization functions."""

import cv2
import numpy as np


class Anonymizer:
    """Apply anonymization to detected face regions."""

    @staticmethod
    def blur(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: float = 0.1) -> np.ndarray:
        """Apply Gaussian blur to face region."""
        h, w = image.shape[:2]

        # Apply padding
        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

        # Extract and blur face region
        face_region = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred

        return image

    @staticmethod
    def black_rectangle(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: float = 0.1) -> np.ndarray:
        """Draw black rectangle over face region."""
        h, w = image.shape[:2]

        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return image

    @staticmethod
    def color_fill(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: str = "#000000", padding: float = 0.1) -> np.ndarray:
        """Fill face region with solid color."""
        h, w = image.shape[:2]

        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

        # Parse hex color to BGR
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

        cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), -1)
        return image

    @staticmethod
    def apply(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, mode: str = "blur", color: str = "#000000", padding: float = 0.1) -> np.ndarray:
        """Apply anonymization based on mode."""
        if mode == "blur":
            return Anonymizer.blur(image, x1, y1, x2, y2, padding)
        elif mode == "black":
            return Anonymizer.black_rectangle(image, x1, y1, x2, y2, padding)
        elif mode == "color":
            return Anonymizer.color_fill(image, x1, y1, x2, y2, color, padding)
        else:
            return Anonymizer.blur(image, x1, y1, x2, y2, padding)
```

**Step 3: Create python/processing/pipeline.py**

```python
"""Video processing pipeline using Ultralytics YOLO."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import av
import numpy as np
from ultralytics import YOLO

from .anonymizer import Anonymizer


@dataclass
class ProcessingStats:
    """Statistics for a processed video."""
    total_frames: int = 0
    processed_frames: int = 0
    faces_detected: int = 0
    processing_time: float = 0.0
    average_fps: float = 0.0
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class ProcessingJob:
    """A video processing job."""
    id: str
    input_path: str
    output_path: str
    model: str = "yolov8m-face"
    mode: str = "blur"
    color: str = "#000000"
    confidence: float = 0.3
    padding: float = 0.1


class VideoProcessor:
    """Process videos to anonymize faces."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: dict[str, YOLO] = {}
        self.current_job: Optional[ProcessingJob] = None
        self.is_cancelled = False

    def load_model(self, model_name: str) -> YOLO:
        """Load a YOLO model, caching for reuse."""
        if model_name not in self.models:
            model_path = self.models_dir / f"{model_name}.pt"
            if not model_path.exists():
                # Try to download from Ultralytics hub or use built-in
                print(f"Loading model: {model_name}")
                self.models[model_name] = YOLO(str(model_path))
            else:
                self.models[model_name] = YOLO(str(model_path))
        return self.models[model_name]

    def process(
        self,
        job: ProcessingJob,
        progress_callback: Optional[Callable[[int, int, int, float], None]] = None,
        resume_frame: int = 0,
    ) -> ProcessingStats:
        """
        Process a video file, anonymizing all detected faces.

        Args:
            job: The processing job configuration
            progress_callback: Called with (frame, total_frames, faces, fps)
            resume_frame: Frame number to resume from (for crash recovery)

        Returns:
            ProcessingStats with final statistics
        """
        self.current_job = job
        self.is_cancelled = False
        stats = ProcessingStats()

        start_time = time.time()
        model = self.load_model(job.model)

        # Open input video
        input_container = av.open(job.input_path)
        input_stream = input_container.streams.video[0]
        stats.total_frames = input_stream.frames or 0

        # Seek to resume point
        if resume_frame > 0:
            timestamp = int(resume_frame / input_stream.average_rate * av.time_base)
            input_container.seek(timestamp)
            stats.processed_frames = resume_frame

        # Open output video
        output_container = av.open(job.output_path, mode='w')
        output_stream = output_container.add_stream(
            codec_name=input_stream.codec_context.name,
            rate=input_stream.average_rate,
        )
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        output_stream.pix_fmt = input_stream.pix_fmt

        frame_count = resume_frame
        fps_start = time.time()
        fps_frames = 0
        current_fps = 0.0

        try:
            for frame in input_container.decode(video=0):
                if self.is_cancelled:
                    break

                # Convert to numpy for processing
                img = frame.to_ndarray(format='bgr24')

                # Detect faces
                results = model.predict(img, conf=job.confidence, verbose=False)

                # Anonymize each detected face
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        stats.faces_detected += 1
                        img = Anonymizer.apply(
                            img, x1, y1, x2, y2,
                            mode=job.mode,
                            color=job.color,
                            padding=job.padding,
                        )

                # Encode output frame
                out_frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                out_frame.pts = frame.pts

                for packet in output_stream.encode(out_frame):
                    output_container.mux(packet)

                frame_count += 1
                fps_frames += 1
                stats.processed_frames = frame_count

                # Calculate FPS every second
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_start = time.time()

                # Progress callback
                if progress_callback and frame_count % 10 == 0:
                    progress_callback(
                        frame_count,
                        stats.total_frames,
                        stats.faces_detected,
                        current_fps,
                    )

        finally:
            # Flush encoder
            for packet in output_stream.encode():
                output_container.mux(packet)

            output_container.close()
            input_container.close()

        stats.processing_time = time.time() - start_time
        stats.average_fps = stats.processed_frames / stats.processing_time if stats.processing_time > 0 else 0

        return stats

    def cancel(self):
        """Cancel current processing job."""
        self.is_cancelled = True

    def save_report(self, job: ProcessingJob, stats: ProcessingStats, output_dir: str):
        """Save processing report as JSON."""
        report_path = Path(output_dir) / f"{Path(job.input_path).stem}_report.json"

        report = {
            "file": job.input_path,
            "output": job.output_path,
            "status": "completed" if not self.is_cancelled else "cancelled",
            "stats": {
                "total_frames": stats.total_frames,
                "processed_frames": stats.processed_frames,
                "faces_detected": stats.faces_detected,
                "processing_time_seconds": round(stats.processing_time, 2),
                "average_fps": round(stats.average_fps, 1),
            },
            "settings": {
                "model": job.model,
                "mode": job.mode,
                "confidence": job.confidence,
                "padding": job.padding,
            },
            "warnings": stats.warnings,
            "errors": stats.errors,
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return str(report_path)
```

**Step 4: Test processing module imports**

Run: `cd python && python -c "from processing import VideoProcessor, Anonymizer; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add python/processing/
git commit -m "feat: implement video processing pipeline with YOLO face detection"
```

---

## Phase 3: Integration

### Task 7: Python Process Manager

**Files:**
- Create: `src/main-process/pythonServer.ts`
- Modify: `src/main.ts`

**Step 1: Create src/main-process/pythonServer.ts**

```typescript
import { spawn, ChildProcess } from 'node:child_process';
import path from 'node:path';
import { app } from 'electron';
import log from 'electron-log/main';
import { EventEmitter } from 'node:events';

const PYTHON_PORT = 8420;
const MAX_RESTART_ATTEMPTS = 3;
const HEALTH_CHECK_INTERVAL = 1000;
const HEALTH_CHECK_TIMEOUT = 30000;

export class PythonServer extends EventEmitter {
  private process: ChildProcess | null = null;
  private restartAttempts = 0;
  private isShuttingDown = false;

  get pythonDir(): string {
    return app.isPackaged
      ? path.join(process.resourcesPath, 'python')
      : path.join(app.getAppPath(), 'python');
  }

  get venvPython(): string {
    const venvPath = path.join(app.getPath('userData'), '.venv');
    return process.platform === 'win32'
      ? path.join(venvPath, 'Scripts', 'python.exe')
      : path.join(venvPath, 'bin', 'python');
  }

  async start(): Promise<void> {
    if (this.process) {
      log.warn('Python server already running');
      return;
    }

    log.info('Starting Python server...');

    const serverScript = path.join(this.pythonDir, 'server.py');

    this.process = spawn(this.venvPython, [serverScript], {
      cwd: this.pythonDir,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    this.process.stdout?.on('data', (data: Buffer) => {
      log.info(`[Python] ${data.toString().trim()}`);
    });

    this.process.stderr?.on('data', (data: Buffer) => {
      log.error(`[Python] ${data.toString().trim()}`);
    });

    this.process.on('exit', (code, signal) => {
      log.info(`Python server exited (code: ${code}, signal: ${signal})`);
      this.process = null;

      if (!this.isShuttingDown && this.restartAttempts < MAX_RESTART_ATTEMPTS) {
        log.warn(`Attempting restart ${this.restartAttempts + 1}/${MAX_RESTART_ATTEMPTS}`);
        this.restartAttempts++;
        setTimeout(() => this.start(), 1000);
      } else if (this.restartAttempts >= MAX_RESTART_ATTEMPTS) {
        this.emit('fatal', 'Python server failed to stay running');
      }
    });

    this.process.on('error', (err) => {
      log.error('Python server error:', err);
      this.emit('error', err.message);
    });

    // Wait for server to be ready
    await this.waitForReady();
    this.restartAttempts = 0;
    this.emit('ready');
  }

  private async waitForReady(): Promise<void> {
    const startTime = Date.now();

    while (Date.now() - startTime < HEALTH_CHECK_TIMEOUT) {
      try {
        const response = await fetch(`http://127.0.0.1:${PYTHON_PORT}/health`);
        if (response.ok) {
          log.info('Python server is ready');
          return;
        }
      } catch {
        // Server not ready yet
      }
      await new Promise((resolve) => setTimeout(resolve, HEALTH_CHECK_INTERVAL));
    }

    throw new Error('Python server failed to start within timeout');
  }

  async stop(): Promise<void> {
    this.isShuttingDown = true;

    if (!this.process) {
      return;
    }

    log.info('Stopping Python server...');

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        log.warn('Force killing Python server');
        this.process?.kill('SIGKILL');
        resolve();
      }, 5000);

      this.process!.once('exit', () => {
        clearTimeout(timeout);
        resolve();
      });

      this.process!.kill('SIGTERM');
    });
  }

  async getStatus(): Promise<{ running: boolean; device?: string }> {
    if (!this.process) {
      return { running: false };
    }

    try {
      const response = await fetch(`http://127.0.0.1:${PYTHON_PORT}/health`);
      const data = await response.json();
      return {
        running: true,
        device: data.device?.name,
      };
    } catch {
      return { running: false };
    }
  }
}

export const pythonServer = new PythonServer();
```

**Step 2: Update src/main.ts to manage Python lifecycle**

Add after createWindow function:

```typescript
import { pythonServer } from './main-process/pythonServer';

// Add to createWindow function, after registerWindowHandlers:
pythonServer.on('ready', () => {
  mainWindow?.webContents.send(IPC_CHANNELS.PYTHON_READY);
});

pythonServer.on('error', (error: string) => {
  mainWindow?.webContents.send(IPC_CHANNELS.PYTHON_ERROR, error);
});

// Add to app.whenReady():
app.whenReady().then(async () => {
  createWindow();
  await pythonServer.start();
});

// Add before app.quit():
app.on('before-quit', async () => {
  await pythonServer.stop();
});
```

**Step 3: Add Python status handler**

Add to src/main.ts:

```typescript
ipcMain.handle(IPC_CHANNELS.PYTHON_STATUS, async () => {
  return pythonServer.getStatus();
});

ipcMain.handle(IPC_CHANNELS.PYTHON_RESTART, async () => {
  await pythonServer.stop();
  await pythonServer.start();
  return { status: 'restarted' };
});
```

**Step 4: Test Python server starts with Electron**

Run: `yarn start`
Expected: Console shows "Python server is ready", UI shows GPU info

**Step 5: Commit**

```bash
git add src/main-process/pythonServer.ts src/main.ts
git commit -m "feat: add Python server process manager with auto-restart"
```

---

### Task 8: Virtual Environment Setup

**Files:**
- Create: `src/main-process/virtualEnvironment.ts`
- Create: `scripts/downloadUV.js`

**Step 1: Create scripts/downloadUV.js**

```javascript
import axios from 'axios';
import fs from 'fs-extra';
import path from 'path';
import * as tar from 'tar';
import extractZip from 'extract-zip';

const UV_VERSION = '0.4.18';
const BASE_URL = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}`;

const platforms = {
  win32: { file: 'uv-x86_64-pc-windows-msvc.zip', folder: 'win', exe: 'uv.exe' },
  darwin: { file: 'uv-aarch64-apple-darwin.tar.gz', folder: 'macos', exe: 'uv' },
  linux: { file: 'uv-x86_64-unknown-linux-gnu.tar.gz', folder: 'linux', exe: 'uv' },
};

async function download(platform) {
  const config = platforms[platform];
  if (!config) {
    console.error(`Unsupported platform: ${platform}`);
    return;
  }

  const outputDir = path.join('./assets/uv', config.folder);
  await fs.ensureDir(outputDir);

  console.log(`Downloading uv ${UV_VERSION} for ${platform}...`);

  const response = await axios.get(`${BASE_URL}/${config.file}`, {
    responseType: 'arraybuffer',
  });

  const tempFile = path.join('./assets', config.file);
  await fs.writeFile(tempFile, response.data);

  console.log(`Extracting to ${outputDir}...`);

  if (config.file.endsWith('.tar.gz')) {
    await tar.extract({ file: tempFile, C: outputDir, strip: 1 });
  } else {
    await extractZip(tempFile, { dir: path.resolve(outputDir) });
  }

  await fs.unlink(tempFile);
  console.log(`Done: ${platform}`);
}

const arg = process.argv[2];

if (arg === 'all') {
  for (const platform of Object.keys(platforms)) {
    await download(platform);
  }
} else {
  await download(process.platform);
}
```

**Step 2: Create src/main-process/virtualEnvironment.ts**

```typescript
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs/promises';
import { app } from 'electron';
import log from 'electron-log/main';
import { EventEmitter } from 'node:events';

export class VirtualEnvironment extends EventEmitter {
  get basePath(): string {
    return app.getPath('userData');
  }

  get venvPath(): string {
    return path.join(this.basePath, '.venv');
  }

  get uvPath(): string {
    const uvFolder = app.isPackaged
      ? path.join(process.resourcesPath, 'uv')
      : path.join(app.getAppPath(), 'assets', 'uv');

    switch (process.platform) {
      case 'win32':
        return path.join(uvFolder, 'win', 'uv.exe');
      case 'darwin':
        return path.join(uvFolder, 'macos', 'uv');
      case 'linux':
        return path.join(uvFolder, 'linux', 'uv');
      default:
        throw new Error(`Unsupported platform: ${process.platform}`);
    }
  }

  get pythonPath(): string {
    return process.platform === 'win32'
      ? path.join(this.venvPath, 'Scripts', 'python.exe')
      : path.join(this.venvPath, 'bin', 'python');
  }

  get requirementsPath(): string {
    const resourcesPath = app.isPackaged
      ? process.resourcesPath
      : path.join(app.getAppPath(), 'assets');

    let filename = 'requirements.txt';

    // Use compiled requirements if available
    if (process.platform === 'win32') {
      filename = 'requirements/windows_nvidia.compiled';
    } else if (process.platform === 'darwin') {
      filename = 'requirements/macos.compiled';
    } else {
      filename = 'requirements/linux.compiled';
    }

    return path.join(resourcesPath, filename);
  }

  async exists(): Promise<boolean> {
    try {
      await fs.access(this.venvPath);
      return true;
    } catch {
      return false;
    }
  }

  async create(onProgress?: (message: string) => void): Promise<void> {
    log.info('Creating virtual environment...');
    onProgress?.('Creating Python environment...');

    // Create venv
    await this.runUv(['venv', '--python', '3.12', this.venvPath], onProgress);

    // Install requirements
    onProgress?.('Installing dependencies (this may take a few minutes)...');
    await this.runUv([
      'pip', 'install',
      '-r', this.requirementsPath,
      '--python', this.pythonPath,
    ], onProgress);

    log.info('Virtual environment created successfully');
    onProgress?.('Setup complete!');
  }

  private runUv(args: string[], onProgress?: (message: string) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      log.info(`Running: uv ${args.join(' ')}`);

      const proc = spawn(this.uvPath, args, {
        env: { ...process.env, VIRTUAL_ENV: this.venvPath },
      });

      proc.stdout?.on('data', (data: Buffer) => {
        const msg = data.toString().trim();
        log.info(`[uv] ${msg}`);
        onProgress?.(msg);
      });

      proc.stderr?.on('data', (data: Buffer) => {
        const msg = data.toString().trim();
        log.warn(`[uv] ${msg}`);
      });

      proc.on('exit', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`uv exited with code ${code}`));
        }
      });

      proc.on('error', reject);
    });
  }
}

export const virtualEnvironment = new VirtualEnvironment();
```

**Step 3: Add to package.json scripts**

```json
"download:uv": "node scripts/downloadUV.js"
```

**Step 4: Test uv download**

Run: `yarn download:uv`
Expected: uv binary downloaded to `assets/uv/{platform}/`

**Step 5: Commit**

```bash
git add src/main-process/virtualEnvironment.ts scripts/downloadUV.js
git commit -m "feat: add virtual environment setup with uv"
```

---

## Phase 4: Packaging

### Task 9: Electron Builder Configuration

**Files:**
- Create: `builder.config.ts`
- Update: `package.json`

**Step 1: Create builder.config.ts**

```typescript
import { Configuration } from 'electron-builder';

const config: Configuration = {
  appId: 'com.fraser.app',
  productName: 'Fraser',

  files: [
    'node_modules/**/*',
    'package.json',
    '.vite/**/*',
  ],

  extraResources: [
    { from: './assets/uv', to: 'uv' },
    { from: './assets/UI', to: 'UI' },
    { from: './assets/requirements', to: 'requirements' },
    { from: './python', to: 'python' },
    { from: './renderer', to: 'renderer' },
  ],

  win: {
    icon: './assets/UI/fraser-icon.ico',
    target: 'nsis',
    artifactName: 'Fraser-Setup-${version}.${ext}',
  },

  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    installerIcon: './assets/UI/fraser-icon.ico',
  },

  mac: {
    icon: './assets/UI/fraser-icon.icns',
    target: 'dmg',
    category: 'public.app-category.video',
    artifactName: 'Fraser-${version}.${ext}',
  },

  linux: {
    icon: './assets/UI/fraser-icon.png',
    target: 'AppImage',
    category: 'Video',
    artifactName: 'Fraser-${version}.${ext}',
  },
};

export default config;
```

**Step 2: Update package.json with build scripts**

Add to scripts:

```json
"make": "yarn build && electron-builder --config builder.config.ts",
"make:win": "yarn make --win nsis",
"make:mac": "yarn make --mac dmg",
"make:linux": "yarn make --linux AppImage"
```

Add devDependency:

```json
"electron-builder": "^25.1.8"
```

**Step 3: Create placeholder icon**

Run: `mkdir -p assets/UI && touch assets/UI/fraser-icon.png`

**Step 4: Test build**

Run: `yarn make`
Expected: Build completes, installer in `dist/`

**Step 5: Commit**

```bash
git add builder.config.ts package.json assets/UI/
git commit -m "feat: add electron-builder configuration for cross-platform packaging"
```

---

## Summary

**Total Tasks:** 9 core tasks across 4 phases

**Phase 1 - Scaffolding (Tasks 1-4):**
- Project setup with Electron + Vite + TypeScript
- Main process entry point
- Renderer UI with Fraser theme
- File dialog handlers

**Phase 2 - Python Backend (Tasks 5-6):**
- FastAPI server with health/models endpoints
- Video processing pipeline with YOLO + PyAV

**Phase 3 - Integration (Tasks 7-8):**
- Python process manager with auto-restart
- Virtual environment setup with uv

**Phase 4 - Packaging (Task 9):**
- Electron-builder for Windows/macOS/Linux

---

**Plan complete and saved to `docs/plans/2025-11-26-fraser-implementation.md`.**

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
