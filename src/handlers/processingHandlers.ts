import { ipcMain, BrowserWindow } from 'electron';
import { IPC_CHANNELS, PYTHON_PORT } from '../constants';
import log from 'electron-log/main';

interface QueueItem {
  id: number;
  path: string;
  type: string;
  name: string;
  duration: string;
  progress: number;
  status: string;
}

interface ProcessingRequest {
  queue: QueueItem[];
  outputPath: string;
  settings: {
    model: string;
    mode: string;
    confidence: number;
  };
}

interface JobStatus {
  type: string;
  job_id: string;
  frame?: number;
  total_frames?: number;
  faces_in_frame?: number;
  fps?: number;
  percent?: number;
  stats?: {
    total_frames: number;
    processed_frames: number;
    faces_detected: number;
    processing_time: number;
    average_fps: number;
  };
  output_path?: string;
  error?: string;
}

let isProcessing = false;
let currentJobId: string | null = null;

async function pollJobStatus(jobId: string): Promise<JobStatus | null> {
  try {
    const response = await fetch(`http://localhost:${PYTHON_PORT}/job/${jobId}`);
    if (response.ok) {
      return await response.json();
    }
  } catch {
    // Job not found yet or server error
  }
  return null;
}

async function processItem(
  mainWindow: BrowserWindow,
  item: QueueItem,
  outputPath: string,
  settings: ProcessingRequest['settings']
): Promise<boolean> {
  log.info(`Starting processing: ${item.name}`);

  // Send initial progress
  mainWindow.webContents.send(IPC_CHANNELS.PROCESS_PROGRESS, {
    itemId: item.id,
    progress: 0,
    status: 'processing'
  });

  // Build output file path from directory and item name
  const path = require('path');
  const outputFilePath = path.join(outputPath, item.name);

  // Start processing on Python server
  const response = await fetch(`http://localhost:${PYTHON_PORT}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input_path: item.path,
      output_path: outputFilePath,
      model: settings.model,
      confidence: settings.confidence,
      padding: 0.20,
      detection_resolution: '360p',
      redaction_mode: settings.mode,
      redaction_color: '#000000',
      temporal_buffer: 5,
      generate_audit: true,
      thumbnail_interval: 30
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    mainWindow.webContents.send(IPC_CHANNELS.PROCESS_PROGRESS, {
      itemId: item.id,
      status: 'error',
      error: `Server error: ${response.status} - ${errorText}`
    });
    return false;
  }

  const result = await response.json();
  currentJobId = result.job_id;
  log.info(`Job started: ${currentJobId}`);

  // Poll for progress until complete
  let lastPercent = 0;
  while (isProcessing && currentJobId) {
    await new Promise(resolve => setTimeout(resolve, 500));

    const status = await pollJobStatus(currentJobId);
    if (!status) continue;

    if (status.type === 'progress') {
      const percent = status.percent || 0;
      if (percent !== lastPercent) {
        lastPercent = percent;
        mainWindow.webContents.send(IPC_CHANNELS.PROCESS_PROGRESS, {
          itemId: item.id,
          progress: percent,
          status: 'processing',
          jobId: currentJobId,
          fps: status.fps,
          facesInFrame: status.faces_in_frame
        });
      }
    } else if (status.type === 'completed') {
      log.info(`Job completed: ${currentJobId}`);
      mainWindow.webContents.send(IPC_CHANNELS.PROCESS_PROGRESS, {
        itemId: item.id,
        progress: 100,
        status: 'completed',
        jobId: currentJobId,
        stats: status.stats,
        outputPath: status.output_path
      });
      return true;
    } else if (status.type === 'error') {
      log.error(`Job error: ${status.error}`);
      mainWindow.webContents.send(IPC_CHANNELS.PROCESS_PROGRESS, {
        itemId: item.id,
        status: 'error',
        error: status.error
      });
      return false;
    }
  }

  // Processing was paused
  return false;
}

export function registerProcessingHandlers(mainWindow: BrowserWindow): void {
  ipcMain.handle(IPC_CHANNELS.PROCESS_START, async (_, request: ProcessingRequest) => {
    if (isProcessing) {
      return { success: false, error: 'Already processing' };
    }

    isProcessing = true;
    log.info(`Starting processing queue with ${request.queue.length} items`);

    try {
      for (const item of request.queue) {
        if (!isProcessing) {
          log.info('Processing paused by user');
          break;
        }

        // Skip already completed items
        if (item.status === 'completed') continue;

        const success = await processItem(mainWindow, item, request.outputPath, request.settings);
        if (!success && isProcessing) {
          // Continue to next item even if one fails
          log.warn(`Item failed, continuing: ${item.name}`);
        }
      }

      isProcessing = false;
      currentJobId = null;
      return { success: true };
    } catch (error) {
      isProcessing = false;
      currentJobId = null;
      const message = error instanceof Error ? error.message : 'Unknown error';
      log.error(`Processing error: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle(IPC_CHANNELS.PROCESS_PAUSE, async () => {
    log.info('Pausing processing');
    isProcessing = false;

    // Cancel current job on Python server
    if (currentJobId) {
      try {
        await fetch(`http://localhost:${PYTHON_PORT}/cancel/${currentJobId}`, {
          method: 'POST'
        });
      } catch {
        // Ignore cancel errors
      }
    }

    return { success: true };
  });
}
