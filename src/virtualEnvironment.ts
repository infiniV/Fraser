import pty from 'node-pty';
import path from 'node:path';
import fs from 'node:fs/promises';
import { app } from 'electron';
import log from 'electron-log/main';
import { EventEmitter } from 'node:events';

export class VirtualEnvironment extends EventEmitter {
  private uvPty: pty.IPty | undefined;

  get basePath(): string {
    return app.getPath('userData');
  }

  get venvPath(): string {
    return path.join(this.basePath, '.venv');
  }

  get uvPath(): string {
    const platform = process.platform;
    let platformFolder: string;

    switch (platform) {
      case 'win32':
        platformFolder = 'win';
        break;
      case 'darwin':
        platformFolder = 'macos';
        break;
      case 'linux':
        platformFolder = 'linux';
        break;
      default:
        throw new Error(`Unsupported platform: ${platform}`);
    }

    const exe = platform === 'win32' ? 'uv.exe' : 'uv';
    const binPath = app.isPackaged
      ? path.join(process.resourcesPath, 'bin', platformFolder, exe)
      : path.join(app.getAppPath(), 'bin', platformFolder, exe);

    return binPath;
  }

  get pythonPath(): string {
    const platform = process.platform;

    if (platform === 'win32') {
      return path.join(this.venvPath, 'Scripts', 'python.exe');
    } else {
      return path.join(this.venvPath, 'bin', 'python');
    }
  }

  get requirementsPath(): string {
    const reqPath = app.isPackaged
      ? path.join(process.resourcesPath, 'python', 'requirements.txt')
      : path.join(app.getAppPath(), 'python', 'requirements.txt');

    return reqPath;
  }

  async exists(): Promise<boolean> {
    try {
      await fs.access(this.venvPath);
      await fs.access(this.pythonPath);
      return true;
    } catch {
      return false;
    }
  }

  async create(onProgress?: (message: string) => void): Promise<void> {
    const progressCallback = (message: string) => {
      log.info(message);
      if (onProgress) {
        onProgress(message);
      }
    };

    try {
      const venvExists = await this.exists();

      if (!venvExists) {
        progressCallback('Creating virtual environment...');
        await this.runUvPty(['venv', this.venvPath], progressCallback);
      } else {
        progressCallback('Virtual environment exists, checking dependencies...');
      }

      // Always run pip install to ensure new dependencies are installed
      // uv is fast and skips already-installed packages
      progressCallback('Installing/updating requirements...');

      const requirementsExists = await fs
        .access(this.requirementsPath)
        .then(() => true)
        .catch(() => false);

      if (requirementsExists) {
        await this.runUvPty(
          ['pip', 'install', '-r', this.requirementsPath],
          progressCallback
        );
      } else {
        progressCallback('No requirements.txt found, skipping package installation');
      }

      progressCallback('Virtual environment setup complete');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      log.error('Failed to create virtual environment:', message);
      throw new Error(`Failed to create virtual environment: ${message}`);
    } finally {
      // Clean up PTY
      if (this.uvPty) {
        this.uvPty.kill();
        this.uvPty = undefined;
      }
    }
  }

  private async runUvPty(
    args: string[],
    onProgress?: (message: string) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const uvPath = this.uvPath;
      const fullCommand = `"${uvPath}" ${args.join(' ')}`;

      log.info(`Running uv command: ${fullCommand}`);

      // Use node-pty for TTY support which enables progress output
      this.uvPty = pty.spawn(uvPath, args, {
        name: 'xterm-256color',
        cols: 120,
        rows: 30,
        cwd: this.basePath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          VIRTUAL_ENV: this.venvPath,
          FORCE_COLOR: '1',
          TERM: 'xterm-256color',
        } as Record<string, string>,
      });

      let output = '';

      this.uvPty.onData((data) => {
        output += data;

        // Clean ANSI codes for logging but show raw for progress parsing
        const cleanLine = data.replace(/\x1b\[[0-9;]*m/g, '').trim();

        if (cleanLine) {
          log.info(`uv: ${cleanLine}`);
          if (onProgress) {
            // Send meaningful lines to progress callback
            const lines = cleanLine.split('\n').filter(l => l.trim());
            lines.forEach(line => {
              onProgress(line);
            });
          }
        }
      });

      this.uvPty.onExit(({ exitCode }) => {
        if (exitCode === 0) {
          resolve();
        } else {
          const error = `uv process exited with code ${exitCode}\nOutput: ${output}`;
          log.error(error);
          reject(new Error(error));
        }
      });
    });
  }
}

export const virtualEnvironment = new VirtualEnvironment();
