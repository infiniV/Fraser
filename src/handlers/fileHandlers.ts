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
