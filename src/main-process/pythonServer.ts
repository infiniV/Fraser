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
    const serverPath = path.join(this.pythonDir, 'server.py');

    try {
      this.process = spawn(this.venvPython, [serverPath], {
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      this.process.stdout?.on('data', (data) => {
        log.info(`[Python] ${data.toString().trim()}`);
      });

      this.process.stderr?.on('data', (data) => {
        log.error(`[Python] ${data.toString().trim()}`);
      });

      this.process.on('exit', (code, signal) => {
        log.info(`Python server exited with code ${code}, signal ${signal}`);
        this.process = null;

        if (!this.isShuttingDown) {
          if (this.restartAttempts < MAX_RESTART_ATTEMPTS) {
            this.restartAttempts++;
            log.warn(`Restarting Python server (attempt ${this.restartAttempts}/${MAX_RESTART_ATTEMPTS})`);
            setTimeout(() => {
              this.start().catch((err) => {
                log.error('Failed to restart Python server:', err);
                this.emit('error', err);
              });
            }, 1000);
          } else {
            log.error('Max restart attempts reached, giving up');
            this.emit('fatal', new Error('Python server crashed too many times'));
          }
        }
      });

      this.process.on('error', (err) => {
        log.error('Python server process error:', err);
        this.emit('error', err);
      });

      await this.waitForReady();
      this.restartAttempts = 0;
      log.info('Python server is ready');
      this.emit('ready');
    } catch (error) {
      log.error('Failed to start Python server:', error);
      this.process = null;
      throw error;
    }
  }

  private async waitForReady(): Promise<void> {
    const startTime = Date.now();
    const healthUrl = `http://127.0.0.1:${PYTHON_PORT}/health`;

    while (Date.now() - startTime < HEALTH_CHECK_TIMEOUT) {
      try {
        const response = await fetch(healthUrl, {
          method: 'GET',
          signal: AbortSignal.timeout(2000),
        });

        if (response.ok) {
          log.info('Python server health check passed');
          return;
        }
      } catch (error) {
        // Server not ready yet, continue polling
      }

      await new Promise((resolve) => setTimeout(resolve, HEALTH_CHECK_INTERVAL));
    }

    throw new Error(`Python server failed to start within ${HEALTH_CHECK_TIMEOUT}ms`);
  }

  async stop(): Promise<void> {
    if (!this.process) {
      log.info('Python server not running');
      return;
    }

    log.info('Stopping Python server...');
    this.isShuttingDown = true;

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        log.warn('Python server did not stop gracefully, killing...');
        this.process?.kill('SIGKILL');
      }, 5000);

      this.process?.once('exit', () => {
        clearTimeout(timeout);
        this.process = null;
        this.isShuttingDown = false;
        log.info('Python server stopped');
        resolve();
      });

      this.process?.kill('SIGTERM');
    });
  }

  async getStatus(): Promise<{ running: boolean; device?: string }> {
    if (!this.process) {
      return { running: false };
    }

    try {
      const healthUrl = `http://127.0.0.1:${PYTHON_PORT}/health`;
      const response = await fetch(healthUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(2000),
      });

      if (response.ok) {
        const data = await response.json() as { device?: string };
        return {
          running: true,
          device: data.device,
        };
      }
    } catch (error) {
      log.error('Failed to get Python server status:', error);
    }

    return { running: false };
  }
}

export const pythonServer = new PythonServer();
