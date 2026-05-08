import { defineConfig, devices } from '@playwright/test'

const host = process.env.PLAYWRIGHT_HOST ?? '127.0.0.1'
const port = Number(process.env.PLAYWRIGHT_PORT ?? 51973)
const baseURL = `http://${host}:${port}`

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  use: { baseURL, trace: 'on-first-retry' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }]
})
