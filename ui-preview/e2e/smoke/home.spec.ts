import { expect, test, type Page } from '@playwright/test'

const FEATURE_FLAG_OVERRIDES_STORAGE_KEY = 'mesh-llm-ui-preview:feature-flags:v1'
const ENABLE_CONFIGURATION_PAGE_OVERRIDE = JSON.stringify({ global: { newConfigurationPage: true } })

async function enableConfigurationPage(page: Page) {
  await page.addInitScript(({ key, value }) => window.localStorage.setItem(key, value), {
    key: FEATURE_FLAG_OVERRIDES_STORAGE_KEY,
    value: ENABLE_CONFIGURATION_PAGE_OVERRIDE
  })
}

test('app smoke navigation hides gated configuration by default', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Your private mesh' })).toBeVisible()
  await expect(
    page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Configuration', exact: true })
  ).toBeHidden()

  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Chat', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page.goto('/configuration/defaults')
  await expect(page.getByRole('heading', { name: 'Configuration is gated' })).toBeVisible()
})

test('configuration blocks in-app navigation when unsaved', async ({ page }) => {
  await enableConfigurationPage(page)
  await page.goto('/chat')
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page
    .getByRole('navigation', { name: 'Primary' })
    .getByRole('link', { name: 'Configuration', exact: true })
    .click()
  await expect(page.getByRole('heading', { name: 'Configuration', exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Model Deployment' }).click()

  const carrackSection = page
    .locator('section')
    .filter({ has: page.getByRole('button', { name: 'Add model to carrack' }) })
  await carrackSection.getByText('pooled', { exact: true }).click()

  await page.goBack()
  await expect(page.getByRole('dialog', { name: 'Unsaved configuration' })).toBeVisible()
  await page.getByRole('button', { name: 'Stay' }).click()
  await expect(page.getByRole('heading', { name: 'Configuration', exact: true })).toBeVisible()

  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Chat', exact: true }).click()
  await expect(page.getByRole('dialog', { name: 'Unsaved configuration' })).toBeVisible()
  await page.getByRole('button', { name: 'Leave page' }).click()
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()
})

test('configuration shows native browser dialog when reloading unsaved changes', async ({ page }) => {
  await enableConfigurationPage(page)
  await page.goto('/chat')
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()

  await page
    .getByRole('navigation', { name: 'Primary' })
    .getByRole('link', { name: 'Configuration', exact: true })
    .click()
  await expect(page.getByRole('heading', { name: 'Configuration', exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Model Deployment' }).click()

  const carrackSection = page
    .locator('section')
    .filter({ has: page.getByRole('button', { name: 'Add model to carrack' }) })
  await carrackSection.getByText('pooled', { exact: true }).click()

  const dialogPromise = page.waitForEvent('dialog')
  const reloadPromise = page.reload({ waitUntil: 'domcontentloaded' })
  const dialog = await dialogPromise

  expect(dialog.type()).toBe('beforeunload')
  await dialog.accept()
  await reloadPromise
  await expect(page.getByRole('heading', { name: 'Configuration', exact: true })).toBeVisible()
})
