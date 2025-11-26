import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import log from 'electron-log/main';
import { registerFileHandlers } from './handlers/fileHandlers';
import { registerProcessingHandlers } from './handlers/processingHandlers';
import { registerQueueHandlers } from './handlers/queueHandlers';
import { IPC_CHANNELS } from './constants';
import { virtualEnvironment } from './virtualEnvironment';
import { pythonServer } from './main-process/pythonServer';

log.initialize();
log.info(`Starting Fraser v${app.getVersion()}`);

let mainWindow: BrowserWindow | null = null;
let isSetupComplete = false;

function sendToRenderer(channel: string, ...args: unknown[]): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, ...args);
  }
}

async function setupPythonEnvironment(): Promise<void> {
  try {
    sendToRenderer(IPC_CHANNELS.SETUP_PROGRESS, 'Checking Python environment...');

    const venvExists = await virtualEnvironment.exists();

    if (!venvExists) {
      sendToRenderer(IPC_CHANNELS.SETUP_PROGRESS, 'Setting up Python environment (first run)...');
      await virtualEnvironment.create((message) => {
        sendToRenderer(IPC_CHANNELS.SETUP_PROGRESS, message);
      });
    }

    sendToRenderer(IPC_CHANNELS.SETUP_PROGRESS, 'Starting Python server...');
    await pythonServer.start();

    isSetupComplete = true;
    sendToRenderer(IPC_CHANNELS.PYTHON_READY);
    log.info('Python environment setup complete');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.error('Failed to setup Python environment:', message);
    sendToRenderer(IPC_CHANNELS.PYTHON_ERROR, message);
  }
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    minWidth: 800,
    minHeight: 600,
    resizable: true,
    backgroundColor: '#0a0a0a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    frame: process.platform === 'darwin',
  });

  // Register handlers
  registerFileHandlers(mainWindow);
  registerProcessingHandlers(mainWindow);
  registerQueueHandlers();

  // Window control handlers
  ipcMain.on(IPC_CHANNELS.WINDOW_MINIMIZE, () => {
    mainWindow?.minimize();
  });

  ipcMain.on(IPC_CHANNELS.WINDOW_MAXIMIZE, () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
  });

  ipcMain.on(IPC_CHANNELS.WINDOW_CLOSE, () => {
    mainWindow?.close();
  });

  // Python status handler
  ipcMain.handle(IPC_CHANNELS.PYTHON_STATUS, async () => {
    return pythonServer.getStatus();
  });

  // Load renderer
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../../renderer/index.html'));
  }

  // Start Python setup after window is ready
  mainWindow.webContents.once('did-finish-load', () => {
    setupPythonEnvironment();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Setup python server event handlers
pythonServer.on('ready', () => {
  sendToRenderer(IPC_CHANNELS.PYTHON_READY);
});

pythonServer.on('error', (error: Error) => {
  sendToRenderer(IPC_CHANNELS.PYTHON_ERROR, error.message);
});

pythonServer.on('fatal', (error: Error) => {
  sendToRenderer(IPC_CHANNELS.PYTHON_ERROR, `Fatal: ${error.message}`);
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  pythonServer.stop().finally(() => {
    app.quit();
  });
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('before-quit', async () => {
  await pythonServer.stop();
});
