import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

const routes = [
  { path: '/', name: 'network dashboard' },
  { path: '/chat', name: 'chat workspace' },
  { path: '/configuration', name: 'gated configuration route' }
] as const

for (const route of routes) {
  test(`${route.name} has no detectable a11y violations`, async ({ page }) => {
    await page.goto(route.path)
    const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
    expect(results.violations).toEqual([])
  })
}
