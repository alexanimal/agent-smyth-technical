import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';
import pkg from './package.json'

// https://vitejs.dev/config/
export default defineConfig({
  base: process.env.NODE_ENV === 'production' ? `/${pkg.name}/` : '/',
  plugins: [
    react(),
    tsconfigPaths()
  ],
});
