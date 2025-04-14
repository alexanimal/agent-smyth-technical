import { sentryVitePlugin } from "@sentry/vite-plugin";
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';
import pkg from './package.json'

// https://vitejs.dev/config/
export default defineConfig({
  base: process.env.NODE_ENV === 'production' ? `/${pkg.name}/` : '/',
  plugins: [react(), tsconfigPaths(), sentryVitePlugin({
    org: "dca-ze",
    project: "agent-smyth"
  })],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'markdown': ['react-markdown', 'react-syntax-highlighter'],
          'ui-components': [
            // UI component paths will go here
          ],
          'message-core': [
            // Message related component paths
          ],
          'message-advanced': [
            // Advanced message components with viewpoints
          ]
        }
      }
    },

    sourcemap: true
  }
});
