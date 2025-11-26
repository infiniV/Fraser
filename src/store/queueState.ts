import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs/promises';
import log from 'electron-log/main';

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

interface QueueState {
  version: number;
  lastUpdated: string;
  outputPath: string;
  settings: {
    model: string;
    mode: string;
    confidence: number;
  };
  queue: QueueItem[];
}

const STATE_VERSION = 1;

class QueueStateManager {
  private statePath: string;
  private state: QueueState | null = null;

  constructor() {
    this.statePath = path.join(app.getPath('userData'), 'queue-state.json');
  }

  async load(): Promise<QueueState | null> {
    try {
      const data = await fs.readFile(this.statePath, 'utf-8');
      this.state = JSON.parse(data);

      if (this.state && this.state.version !== STATE_VERSION) {
        log.warn('Queue state version mismatch, ignoring saved state');
        return null;
      }

      log.info(`Loaded queue state: ${this.state?.queue.length || 0} items`);
      return this.state;
    } catch (error) {
      // No saved state or invalid JSON
      return null;
    }
  }

  async save(state: Omit<QueueState, 'version' | 'lastUpdated'>): Promise<void> {
    const fullState: QueueState = {
      version: STATE_VERSION,
      lastUpdated: new Date().toISOString(),
      ...state
    };

    try {
      await fs.writeFile(this.statePath, JSON.stringify(fullState, null, 2));
      this.state = fullState;
      log.info(`Saved queue state: ${state.queue.length} items`);
    } catch (error) {
      log.error('Failed to save queue state:', error);
    }
  }

  async clear(): Promise<void> {
    try {
      await fs.unlink(this.statePath);
      this.state = null;
      log.info('Cleared queue state');
    } catch {
      // File doesn't exist, ignore
    }
  }

  getState(): QueueState | null {
    return this.state;
  }
}

export const queueState = new QueueStateManager();
