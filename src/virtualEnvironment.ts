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
      : path.join(__dirname, '..', 'bin', platformFolder, exe);

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
      : path.join(__dirname, '..', 'python', 'requirements.txt');

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
      progressCallback('Creating virtual environment...');

      const venvExists = await this.exists();
      if (venvExists) {
        progressCallback('Virtual environment already exists');
        return;
      }

      await this.runUv(['venv', this.venvPath], progressCallback);

      progressCallback('Installing requirements...');

      const requirementsExists = await fs
        .access(this.requirementsPath)
        .then(() => true)
        .catch(() => false);

      if (requirementsExists) {
        await this.runUv(
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
    }
  }

  private async runUv(
    args: string[],
    onProgress?: (message: string) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const uvPath = this.uvPath;

      log.info(`Running uv command: ${uvPath} ${args.join(' ')}`);

      const child = spawn(uvPath, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
        },
      });

      let stdout = '';
      let stderr = '';

      child.stdout?.on('data', (data: Buffer) => {
        const message = data.toString();
        stdout += message;
        log.info(`uv stdout: ${message.trim()}`);
        if (onProgress) {
          onProgress(message.trim());
        }
      });

      child.stderr?.on('data', (data: Buffer) => {
        const message = data.toString();
        stderr += message;
        log.info(`uv stderr: ${message.trim()}`);
        if (onProgress) {
          onProgress(message.trim());
        }
      });

      child.on('error', (error) => {
        log.error('Failed to spawn uv process:', error);
        reject(new Error(`Failed to spawn uv: ${error.message}`));
      });

      child.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          const error = `uv process exited with code ${code}\nstdout: ${stdout}\nstderr: ${stderr}`;
          log.error(error);
          reject(new Error(error));
        }
      });
    });
  }
}

export const virtualEnvironment = new VirtualEnvironment();
