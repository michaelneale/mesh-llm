import path from 'node:path'
import { fileURLToPath } from 'node:url'

const dirname = path.dirname(fileURLToPath(import.meta.url))

export default {
  resolve: {
    alias: {
      '@': path.resolve(dirname, './src'),
    },
  },
  test: {
    environment: 'node',
    globals: true,
    setupFiles: ['./src/lib/test/setup.ts'],
    include: [
      'src/features/network/api/status-adapter.test.ts',
      'src/features/network/components/MeshViz.helpers.test.ts',
      'src/features/network/components/MeshVizNodeHoverCard.test.tsx',
      'src/features/network/components/PeersTable.test.tsx',
      'src/features/drawers/components/NodeDrawer.test.tsx',
    ],
  },
}
