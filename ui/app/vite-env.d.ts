/// <reference types="vite/client" />

declare module '@vitejs/plugin-react' {
  import type { Plugin } from 'vite';
  
  interface Options {
    include?: string | RegExp | (string | RegExp)[];
    exclude?: string | RegExp | (string | RegExp)[];
    babel?: Record<string, unknown>;
    jsxRuntime?: 'automatic' | 'classic';
    jsxImportSource?: string;
  }
  
  function reactPlugin(options?: Options): Plugin;
  export default reactPlugin;
}

declare module '@tailwindcss/vite' {
  import type { Plugin } from 'vite';
  
  interface TailwindOptions {
    configFile?: string;
    config?: Record<string, unknown>;
  }
  
  function tailwindPlugin(options?: TailwindOptions): Plugin;
  export default tailwindPlugin;
}
