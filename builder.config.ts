import { Configuration } from 'electron-builder';

const config: Configuration = {
  appId: 'com.fraser.app',
  productName: 'Fraser',

  files: [
    'node_modules/**/*',
    'package.json',
    '.vite/**/*',
  ],

  extraResources: [
    { from: './assets/uv', to: 'uv' },
    { from: './assets/UI', to: 'UI' },
    { from: './assets/requirements', to: 'requirements' },
    { from: './python', to: 'python' },
    { from: './renderer', to: 'renderer' },
  ],

  win: {
    icon: './assets/UI/fraser-icon.ico',
    target: ['dir', 'zip'],
    artifactName: 'Fraser-${version}-win.${ext}',
    signAndEditExecutable: false,
  },

  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    installerIcon: './assets/UI/fraser-icon.ico',
  },

  mac: {
    icon: './assets/UI/fraser-icon.icns',
    target: 'dmg',
    category: 'public.app-category.video',
    artifactName: 'Fraser-${version}.${ext}',
  },

  linux: {
    icon: './assets/UI/fraser-icon.png',
    target: 'AppImage',
    category: 'Video',
    artifactName: 'Fraser-${version}.${ext}',
  },
};

export default config;
