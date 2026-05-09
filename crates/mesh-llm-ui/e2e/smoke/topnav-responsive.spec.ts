import { expect, test, type Page } from '@playwright/test'

const BREAKPOINT_WIDTHS = [360, 420, 500, 640, 767, 768, 800, 840, 900, 1023, 1024, 1133, 1280, 1440]

async function readTopNavMetrics(page: Page) {
  return page.evaluate(() => {
    const header = document.querySelector('header')
    const apiTarget = document.querySelector('[aria-label="API target instructions"]')
    const actionsMenu = document.querySelector('[aria-label="Open navigation actions"]')
    const joinButton = document.querySelector('[aria-label="Mesh join and invite instructions"]')
    const themeButton = document.querySelector('[aria-label^="Theme:"]')
    const preferencesButton = document.querySelector('[aria-label="Open interface preferences"]')
    const fullTabLabels = [...document.querySelectorAll('a')].filter((element) =>
      ['Network', 'Chat', 'Configuration'].includes((element.textContent ?? '').trim())
    )

    const isVisible = (element: Element | null): element is Element => {
      if (!element) return false
      const style = window.getComputedStyle(element)
      const rect = element.getBoundingClientRect()

      return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0
    }

    const visibleControls = [
      apiTarget,
      actionsMenu,
      joinButton,
      themeButton,
      preferencesButton,
      ...fullTabLabels
    ].filter(isVisible)
    const controlTops = visibleControls.map((element) => Math.round(element.getBoundingClientRect().top))
    const controlTopSpread = controlTops.length > 0 ? Math.max(...controlTops) - Math.min(...controlTops) : 0

    return {
      actionsMenuVisible: isVisible(actionsMenu),
      apiTargetVisible: isVisible(apiTarget),
      controlTopSpread,
      fullTabLabels: fullTabLabels.filter(isVisible).map((element) => (element.textContent ?? '').trim()),
      headerHeight: header ? Math.round(header.getBoundingClientRect().height) : 0,
      horizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
      joinButtonVisible: isVisible(joinButton),
      preferencesButtonVisible: isVisible(preferencesButton),
      themeButtonVisible: isVisible(themeButton)
    }
  })
}

test.skip('top navigation stays on one row at every responsive breakpoint', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Your private mesh' })).toBeVisible()

  for (const width of BREAKPOINT_WIDTHS) {
    await page.setViewportSize({ width, height: 1133 })
    await page.waitForTimeout(100)

    const metrics = await readTopNavMetrics(page)

    expect(metrics.headerHeight, `${width}px header should remain a single row`).toBeLessThanOrEqual(60)
    expect(metrics.controlTopSpread, `${width}px controls should not split onto separate rows`).toBeLessThanOrEqual(3)
    expect(metrics.horizontalOverflow, `${width}px should not create horizontal document overflow`).toBe(false)

    if (width < 768) {
      expect(metrics.apiTargetVisible, `${width}px compact state hides API target chip`).toBe(false)
      expect(metrics.actionsMenuVisible, `${width}px compact state keeps actions in the menu`).toBe(true)
    } else {
      expect(metrics.apiTargetVisible, `${width}px shows the API target chip`).toBe(true)
      expect(metrics.actionsMenuVisible, `${width}px middle state uses the actions menu`).toBe(true)
    }
  }
})
