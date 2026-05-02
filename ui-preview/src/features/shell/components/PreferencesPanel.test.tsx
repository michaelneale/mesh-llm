import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PreferencesPanel } from '@/features/shell/components/PreferencesPanel'

describe('PreferencesPanel', () => {
  it('offers auto, dark, and light theme choices', async () => {
    const user = userEvent.setup()
    const onThemeChange = vi.fn()

    render(
      <PreferencesPanel
        open
        theme="auto"
        accent="blue"
        density="normal"
        panelStyle="solid"
        onThemeChange={onThemeChange}
        onAccentChange={vi.fn()}
        onDensityChange={vi.fn()}
        onPanelStyleChange={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    expect(screen.getByRole('radio', { name: 'Auto' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Dark' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('radio', { name: 'Light' })).toHaveAttribute('aria-checked', 'false')

    await user.click(screen.getByRole('radio', { name: 'Light' }))
    expect(onThemeChange).toHaveBeenCalledWith('light')
  })

  it('shows normal as the selected default density and exposes larger sparse sizing', async () => {
    const user = userEvent.setup()
    const onDensityChange = vi.fn()

    render(
      <PreferencesPanel
        open
        theme="dark"
        accent="blue"
        density="normal"
        panelStyle="solid"
        onThemeChange={vi.fn()}
        onAccentChange={vi.fn()}
        onDensityChange={onDensityChange}
        onPanelStyleChange={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    expect(screen.getByRole('heading', { name: 'Preferences' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Compact' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('radio', { name: 'Normal' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Sparse' })).toHaveAttribute('aria-checked', 'false')

    await user.click(screen.getByRole('radio', { name: 'Sparse' }))
    expect(onDensityChange).toHaveBeenCalledWith('sparse')

    await user.click(screen.getByRole('radio', { name: 'Compact' }))
    expect(onDensityChange).toHaveBeenCalledWith('compact')
  })

  it('exposes the reversible global panel style tweak', async () => {
    const user = userEvent.setup()
    const onPanelStyleChange = vi.fn()

    render(
      <PreferencesPanel
        open
        theme="dark"
        accent="blue"
        density="normal"
        panelStyle="solid"
        onThemeChange={vi.fn()}
        onAccentChange={vi.fn()}
        onDensityChange={vi.fn()}
        onPanelStyleChange={onPanelStyleChange}
        onClose={vi.fn()}
      />,
    )

    expect(screen.getByRole('radio', { name: 'Solid' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Soft' })).toHaveAttribute('aria-checked', 'false')

    await user.click(screen.getByRole('radio', { name: 'Soft' }))
    expect(onPanelStyleChange).toHaveBeenCalledWith('soft')
  })

  it('omits pink from the accent colour picker', () => {
    render(
      <PreferencesPanel
        open
        theme="dark"
        accent="blue"
        density="normal"
        panelStyle="solid"
        onThemeChange={vi.fn()}
        onAccentChange={vi.fn()}
        onDensityChange={vi.fn()}
        onPanelStyleChange={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    expect(screen.queryByRole('radio', { name: 'Use pink accent' })).not.toBeInTheDocument()
  })
})
