import { spawn } from 'node:child_process'
import { createServer } from 'node:net'

const host = process.env.PLAYWRIGHT_HOST ?? '127.0.0.1'

function findAvailablePort() {
  return new Promise((resolve, reject) => {
    const server = createServer()

    server.once('error', reject)
    server.listen(0, host, () => {
      const address = server.address()
      server.close((error) => {
        if (error) {
          reject(error)
          return
        }

        if (address && typeof address === 'object') {
          resolve(address.port)
          return
        }

        reject(new Error('Unable to reserve a Playwright preview port.'))
      })
    })
  })
}

const port = process.env.PLAYWRIGHT_PORT ?? String(await findAvailablePort())
const child = spawn('playwright', ['test', ...process.argv.slice(2)], {
  env: { ...process.env, PLAYWRIGHT_HOST: host, PLAYWRIGHT_PORT: port },
  shell: process.platform === 'win32',
  stdio: 'inherit',
})

child.once('error', (error) => {
  console.error(error)
  process.exit(1)
})

child.once('exit', (code) => {
  process.exit(code ?? 1)
})
