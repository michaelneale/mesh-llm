/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MESH_LLM_DEBUG_UI?: string
}

declare module '*.svg' {
  const src: string
  export default src
}
