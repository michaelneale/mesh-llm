const path = require('node:path')

const appDir = path.resolve(__dirname, 'crates/mesh-llm-ui/preview')
const testDir = path.resolve(appDir, 'e2e')
const host = process.env.PLAYWRIGHT_HOST ?? '127.0.0.1'
const port = Number(process.env.PLAYWRIGHT_PORT ?? 51973)
const baseURL = `http://${host}:${port}`
const reuseExistingServer = process.env.PLAYWRIGHT_REUSE_EXISTING_SERVER === 'true' && !process.env.CI

module.exports = {
  testDir,
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  use: { baseURL, trace: 'on-first-retry' },
  webServer: {
    command: `npm --prefix "${appDir}" run build && npm --prefix "${appDir}" run preview -- --host ${host} --port ${port} --strictPort`,
    cwd: appDir,
    url: baseURL,
    reuseExistingServer,
    timeout: 120000,
    stderr: 'pipe',
    stdout: 'ignore'
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }]
}
