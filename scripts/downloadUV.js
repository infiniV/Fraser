import axios from 'axios';
import fs from 'fs-extra';
import path from 'path';
import * as tar from 'tar';
import { pipeline } from 'node:stream/promises';
import { createWriteStream } from 'node:fs';
import AdmZip from 'adm-zip';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const UV_VERSION = '0.4.18';
const BASE_URL = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}`;

const platforms = {
  win32: { file: 'uv-x86_64-pc-windows-msvc.zip', folder: 'win', exe: 'uv.exe' },
  darwin: { file: 'uv-aarch64-apple-darwin.tar.gz', folder: 'macos', exe: 'uv' },
  linux: { file: 'uv-x86_64-unknown-linux-gnu.tar.gz', folder: 'linux', exe: 'uv' },
};

async function downloadAndExtract(platform, platformInfo) {
  const url = `${BASE_URL}/${platformInfo.file}`;
  const destDir = path.join(__dirname, '..', 'bin', platformInfo.folder);
  const tempFile = path.join(__dirname, '..', 'bin', platformInfo.file);

  console.log(`Downloading uv for ${platform} from ${url}...`);

  await fs.ensureDir(path.join(__dirname, '..', 'bin'));

  try {
    const response = await axios({
      method: 'get',
      url: url,
      responseType: 'stream',
    });

    await pipeline(response.data, createWriteStream(tempFile));

    console.log(`Extracting to ${destDir}...`);
    await fs.ensureDir(destDir);

    if (platformInfo.file.endsWith('.zip')) {
      const zip = new AdmZip(tempFile);
      zip.extractAllTo(destDir, true);
    } else if (platformInfo.file.endsWith('.tar.gz')) {
      await tar.x({
        file: tempFile,
        cwd: destDir,
      });
    }

    await fs.remove(tempFile);

    const uvBinary = path.join(destDir, 'uv-x86_64-unknown-linux-gnu', platformInfo.exe);
    if (await fs.pathExists(uvBinary)) {
      const finalPath = path.join(destDir, platformInfo.exe);
      await fs.move(uvBinary, finalPath, { overwrite: true });
      await fs.remove(path.join(destDir, 'uv-x86_64-unknown-linux-gnu'));
    }

    const uvBinaryWin = path.join(destDir, 'uv-x86_64-pc-windows-msvc', platformInfo.exe);
    if (await fs.pathExists(uvBinaryWin)) {
      const finalPath = path.join(destDir, platformInfo.exe);
      await fs.move(uvBinaryWin, finalPath, { overwrite: true });
      await fs.remove(path.join(destDir, 'uv-x86_64-pc-windows-msvc'));
    }

    const uvBinaryMac = path.join(destDir, 'uv-aarch64-apple-darwin', platformInfo.exe);
    if (await fs.pathExists(uvBinaryMac)) {
      const finalPath = path.join(destDir, platformInfo.exe);
      await fs.move(uvBinaryMac, finalPath, { overwrite: true });
      await fs.remove(path.join(destDir, 'uv-aarch64-apple-darwin'));
    }

    const finalBinary = path.join(destDir, platformInfo.exe);
    if (platform !== 'win32') {
      await fs.chmod(finalBinary, 0o755);
    }

    console.log(`Successfully downloaded and extracted uv for ${platform}`);
  } catch (error) {
    console.error(`Error downloading uv for ${platform}:`, error.message);
    throw error;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const downloadAll = args.includes('all');

  if (downloadAll) {
    console.log('Downloading uv for all platforms...');
    for (const [platform, platformInfo] of Object.entries(platforms)) {
      await downloadAndExtract(platform, platformInfo);
    }
  } else {
    const currentPlatform = process.platform;
    const platformInfo = platforms[currentPlatform];

    if (!platformInfo) {
      console.error(`Unsupported platform: ${currentPlatform}`);
      process.exit(1);
    }

    await downloadAndExtract(currentPlatform, platformInfo);
  }

  console.log('Done!');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
