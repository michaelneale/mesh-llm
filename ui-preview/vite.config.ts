import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { PluginOption } from 'vite'
import { defineConfig } from 'vitest/config'

const dirname = path.dirname(fileURLToPath(import.meta.url))

function plugins(): PluginOption[] {
  const vitePlugins: PluginOption[] = []
  // The app uses code-defined TanStack routes by default. Keep the file-router generator
  // opt-in until route files are added so typecheck/test never depend on missing stubs.
  if (process.env.TANSTACK_FILE_ROUTER === 'true') {
    vitePlugins.push(tanstackRouter({
      target: 'react',
      autoCodeSplitting: true,
      routesDirectory: './src/app/router/routes',
      generatedRouteTree: './src/routeTree.gen.ts',
    }))
  }
  vitePlugins.push(react(), tailwindcss())
  return vitePlugins
}

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? '/',
  plugins: plugins(),
  resolve: { alias: { '@': path.resolve(dirname, './src') } },
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9337',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/lib/test/setup.ts'],
    css: true,
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
