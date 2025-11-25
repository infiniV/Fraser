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
