import { ipcMain } from 'electron';
import { IPC_CHANNELS } from '../constants';
import { queueState } from '../store/queueState';

interface QueueItem {
  id: number;
  path: string;
  type: string;
  name: string;
  duration: string;
  progress: number;
  status: string;
  jobId?: string;
  faces?: number;
  error?: string;
}

interface SaveQueueRequest {
  queue: QueueItem[];
  outputPath: string;
  settings: {
    model: string;
    mode: string;
    confidence: number;
  };
}

export function registerQueueHandlers(): void {
  ipcMain.handle(IPC_CHANNELS.QUEUE_LOAD, async () => {
    return await queueState.load();
  });

  ipcMain.handle(IPC_CHANNELS.QUEUE_SAVE, async (_, request: SaveQueueRequest) => {
    await queueState.save(request);
    return { success: true };
  });

  ipcMain.handle(IPC_CHANNELS.QUEUE_CLEAR, async () => {
    await queueState.clear();
    return { success: true };
  });
}
