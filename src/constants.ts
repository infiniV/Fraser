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

  // Install/Setup
  INSTALL_PROGRESS: 'install:progress',
  SETUP_PROGRESS: 'setup:progress',

  // Window
  WINDOW_MINIMIZE: 'window:minimize',
  WINDOW_MAXIMIZE: 'window:maximize',
  WINDOW_CLOSE: 'window:close',

  // Processing
  PROCESS_START: 'process:start',
  PROCESS_PAUSE: 'process:pause',
  PROCESS_PROGRESS: 'process:progress',

  // Queue State
  QUEUE_LOAD: 'queue:load',
  QUEUE_SAVE: 'queue:save',
  QUEUE_CLEAR: 'queue:clear',
} as const;

export const PYTHON_PORT = 8420;

export const SUPPORTED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'];
