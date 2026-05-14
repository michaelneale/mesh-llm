import { expect, test } from '@playwright/test'

const meshHeroHeading = /^(Your private mesh|Welcome to the public mesh)$/

test('app smoke navigation shows Network and Chat tabs', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: meshHeroHeading })).toBeVisible()

  await expect(
    page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Network', exact: true })
  ).toHaveAttribute('aria-current', 'page')

  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Chat', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible()
  await expect(
    page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Chat', exact: true })
  ).toHaveAttribute('aria-current', 'page')

  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Network', exact: true }).click()
  await expect(page.getByRole('heading', { name: meshHeroHeading })).toBeVisible()
})

test('Network tab shows peer status and model information', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: meshHeroHeading })).toBeVisible()
  await expect(page.getByText('Serving').first()).toBeVisible()
  await expect(page.getByText(/GB/).first()).toBeVisible()
})
