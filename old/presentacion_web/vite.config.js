import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  build: {
    target: 'es2022',
    modulePreload: false,
    cssCodeSplit: false,
    rolldownOptions: { output: { codeSplitting: false } },
  },
});
