import { defineConfig } from 'vite';
import { resolve } from 'path';
import { builtinModules } from 'module';

export default defineConfig({
  build: {
    outDir: '.vite/build',
    emptyOutDir: true,
    lib: {
      entry: resolve(__dirname, 'src/main.ts'),
      formats: ['cjs'],
      fileName: () => 'main.cjs',
    },
    rollupOptions: {
      external: [
        'electron',
        'electron-store',
        'electron-log',
        'node-pty',
        ...builtinModules,
        ...builtinModules.map(m => `node:${m}`),
      ],
    },
    minify: false,
    commonjsOptions: {
      ignoreDynamicRequires: true,
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
});
