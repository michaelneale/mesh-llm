import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Circle } from 'lucide-react'
import { describe, expect, it, vi } from 'vitest'
import { TabPanel, type TabPanelItem } from '@/components/ui/TabPanel'

const tabItems: TabPanelItem<'alpha' | 'beta'>[] = [
  {
    value: 'alpha',
    label: 'Alpha',
    icon: Circle,
    accessory: <span className="rounded-full border border-border-soft px-[5px] font-mono text-[length:var(--density-type-annotation)]">2</span>,
    content: <div>Alpha panel</div>,
  },
  {
    value: 'beta',
    label: 'Beta',
    accessory: ({ active }) => (active ? <span title="Beta active marker" className="size-[5px] rounded-full bg-warn" /> : null),
    content: <div>Beta panel</div>,
  },
]

describe('TabPanel', () => {
  it('manages tab state when uncontrolled and renders icons plus accessories', async () => {
    const user = userEvent.setup()

    render(<TabPanel ariaLabel="Example tabs" defaultValue="alpha" tabs={tabItems} />)

    expect(screen.getByRole('tab', { name: /alpha2/i })).toHaveAttribute('data-active', 'true')
    expect(screen.getByText('Alpha panel')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Beta' }))

    expect(screen.getByRole('tab', { name: 'Beta' })).toHaveAttribute('data-active', 'true')
    expect(screen.getByTitle('Beta active marker')).toBeInTheDocument()
    expect(screen.getByText('Beta panel')).toBeInTheDocument()
  })

  it('supports controlled state updates', async () => {
    const user = userEvent.setup()
    const handleValueChange = vi.fn()

    render(<TabPanel ariaLabel="Controlled tabs" onValueChange={handleValueChange} tabs={tabItems} value="alpha" />)

    await user.click(screen.getByRole('tab', { name: 'Beta' }))

    expect(handleValueChange).toHaveBeenCalledWith('beta')
    expect(screen.getByRole('tab', { name: /alpha2/i })).toHaveAttribute('data-active', 'true')
  })

  it('uses the first enabled tab when an uncontrolled default tab is disabled', () => {
    render(
      <TabPanel
        ariaLabel="Disabled default tabs"
        defaultValue="beta"
        tabs={[
          { value: 'alpha', label: 'Alpha', content: <div>Alpha panel</div> },
          { value: 'beta', label: 'Beta', content: <div>Beta panel</div>, disabled: true },
        ]}
      />,
    )

    expect(screen.getByRole('tab', { name: 'Alpha' })).toHaveAttribute('data-active', 'true')
    expect(screen.getByRole('tab', { name: 'Beta' })).toBeDisabled()
  })

  it('does not silently select a fallback tab for an invalid controlled value', () => {
    render(
      <TabPanel
        ariaLabel="Controlled disabled tabs"
        tabs={[
          { value: 'alpha', label: 'Alpha', content: <div>Alpha panel</div> },
          { value: 'beta', label: 'Beta', content: <div>Beta panel</div>, disabled: true },
        ]}
        value="beta"
      />,
    )

    expect(screen.getByRole('tab', { name: 'Alpha' })).not.toHaveAttribute('data-active')
    expect(screen.getByRole('tab', { name: 'Beta' })).toBeDisabled()
    expect(screen.getByRole('tab', { name: 'Beta' })).not.toHaveAttribute('data-active')
  })
})
