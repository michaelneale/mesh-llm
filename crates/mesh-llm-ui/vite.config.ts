import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { PluginOption } from 'vite'
import { defineConfig } from 'vitest/config'

const dirname = path.dirname(fileURLToPath(import.meta.url))
const apiTarget = process.env.MESH_UI_API_ORIGIN ?? 'http://127.0.0.1:3131'

/**
 * Stop Vite from bundling the ~21 MB ONNX Runtime WASM blob that ships inside
 * onnxruntime-web (pulled in transitively by @huggingface/transformers).
 *
 * Background: the ORT loader contains literals like
 *   new URL("ort-wasm-simd-threaded.jsep.wasm", import.meta.url)
 * which Vite/Rollup's `assetImportMetaUrl` pass recognises and aggressively
 * copies the .wasm into `dist/assets/` — even when Transformers.js sets
 * `env.backends.onnx.wasm.wasmPaths` to a jsDelivr CDN URL at runtime (which
 * makes the bundled copy completely unused at runtime).
 *
 * Strategy: defeat Vite's pattern matcher by transforming any source that
 * contains the offending pattern. We rewrite
 *   new URL("ort-wasm-*.wasm", import.meta.url)
 * to a semantically-equivalent runtime expression that does NOT match Vite's
 * AST pattern (it requires both args to be the exact literal forms).
 *
 * After this transform, Vite has no static asset reference to follow, so it
 * never emits the WASM. At runtime, Transformers.js's `wasmPaths` CDN setting
 * takes over via `config.locateFile` (see onnxruntime-web/wasm-factory.ts),
 * so behaviour is unchanged from the user's perspective — only the bundle is
 * lighter by ~21 MB.
 *
 * Affects only files that contain `ort-wasm-*` URL literals (the few ORT
 * bundles), so it is safe to run across the whole graph.
 */
function stripBundledOnnxWasm(): PluginOption {
  // Matches both jsep and non-jsep variants. The pattern is intentionally
  // strict about whitespace and quoting so we only touch the ORT loader
  // literals and not any user code that happens to mention "ort-wasm".
  const PATTERN = /new URL\((['"])(ort-wasm[a-zA-Z0-9._-]*\.wasm)\1,\s*import\.meta\.url\)/g
  return {
    name: 'mesh-llm:strip-bundled-onnx-wasm',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('onnxruntime-web')) return null
      if (!PATTERN.test(code)) return null
      // Reset lastIndex because we tested with `.test()` above.
      PATTERN.lastIndex = 0
      // Replace `new URL("ort-wasm-*.wasm", import.meta.url)` with an
      // equivalent runtime expression that doesn't trip Vite's asset emitter.
      // The wrapping IIFE evaluates to the same URL object/string the loader
      // expects; ORT only uses it as a fallback when `locateFile` is not set.
      // Split the filename across a runtime concat so Vite's literal-only
      // asset-emitter pattern matcher does not constant-fold it back into a
      // static URL. We also alias `import.meta` to a binding so it stops
      // being a syntactic match for `import.meta.url`.
      const rewritten = code.replace(PATTERN, (_match, _quote, file) => {
        const half = Math.floor(file.length / 2)
        const a = JSON.stringify(file.slice(0, half))
        const b = JSON.stringify(file.slice(half))
        return `new URL(${a}+${b}, (0,()=>import.meta)().url)`
      })
      return { code: rewritten, map: null }
    }
  }
}

function plugins(): PluginOption[] {
  const vitePlugins: PluginOption[] = []
  // The app uses code-defined TanStack routes by default. Keep the file-router generator
  // opt-in until route files are added so typecheck/test never depend on missing stubs.
  if (process.env.TANSTACK_FILE_ROUTER === 'true') {
    vitePlugins.push(
      tanstackRouter({
        target: 'react',
        autoCodeSplitting: true,
        routesDirectory: './src/app/router/routes',
        generatedRouteTree: './src/routeTree.gen.ts'
      })
    )
  }
  vitePlugins.push(stripBundledOnnxWasm(), react(), tailwindcss())
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
        target: apiTarget,
        changeOrigin: true,
        headers: {
          'User-Agent': 'Mozilla/5.0 mesh-llm-ui-preview-dev-proxy'
        }
      }
    }
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/lib/test/setup.ts'],
    css: true,
    include: ['src/**/*.test.{ts,tsx}']
  }
})
