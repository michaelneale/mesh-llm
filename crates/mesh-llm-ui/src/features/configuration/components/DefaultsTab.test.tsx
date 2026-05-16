import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import { DefaultsTab } from '@/features/configuration/components/DefaultsTab'
import type {
  ConfigurationDefaultsHarnessData,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues
} from '@/features/app-tabs/types'
import { env } from '@/lib/env'
import { RESTART_REQUIRED_TOOLTIP } from '@/features/configuration/components/settings/RestartBadge'

const SHOW_ADVANCED_STORAGE_KEY = `${env.storageNamespace}:configuration-defaults:show-advanced:v1`

const defaultSettings = [
  {
    id: 'runtime-mode',
    categoryId: 'runtime',
    icon: 'cpu',
    label: 'Runtime mode',
    description: 'Controls the standard runtime selection.',
    inheritedLabel: 'Inherited by default placements',
    control: {
      kind: 'choice',
      name: 'runtime_mode',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'manual', label: 'manual' }
      ]
    }
  },
  {
    id: 'advanced-reasoning',
    categoryId: 'advanced',
    icon: 'cog',
    label: 'Reasoning budget',
    description: 'Advanced reasoning control.',
    inheritedLabel: 'Inherited by reasoning-capable placements',
    visibility: 'advanced',
    control: {
      kind: 'range',
      name: 'reasoning_budget',
      value: '128',
      min: 0,
      max: 512,
      step: 32,
      unit: 'tok'
    }
  },
  {
    id: 'advanced-note',
    categoryId: 'advanced',
    icon: 'filter',
    label: 'Advanced note',
    description: 'Extra advanced guidance.',
    inheritedLabel: 'Inherited by advanced defaults',
    visibility: 'advanced',
    control: {
      kind: 'text',
      name: 'advanced_note',
      value: '',
      placeholder: 'Optional note'
    }
  }
] satisfies readonly ConfigurationDefaultsSetting[]

const defaultsData = {
  categories: [
    { id: 'runtime', label: 'Runtime', summary: 'Standard defaults.', help: 'Runtime defaults' },
    { id: 'advanced', label: 'Reasoning', summary: 'Advanced defaults.', help: 'Reasoning defaults' }
  ],
  settings: defaultSettings,
  preview: []
} satisfies ConfigurationDefaultsHarnessData

const dependencySettings = [
  {
    id: 'speculation-mode',
    categoryId: 'speculative-decoding',
    icon: 'brain',
    label: 'Speculation mode',
    description: 'Controls whether draft-model speculation is active.',
    inheritedLabel: 'Inherited by speculative decoding defaults',
    control: {
      kind: 'choice',
      name: 'speculation_mode',
      value: 'ngram',
      presentation: 'segmented',
      options: [
        { value: 'off', label: 'off' },
        { value: 'draft_model', label: 'draft model' },
        { value: 'ngram', label: 'n-gram' }
      ]
    }
  },
  {
    id: 'draft-selection-policy',
    categoryId: 'speculative-decoding',
    icon: 'filter',
    label: 'Draft selection policy',
    description: 'Chooses the draft selection behavior.',
    inheritedLabel: 'Only available when draft-model speculation is enabled',
    dependsOn: {
      settingId: 'speculation-mode',
      condition: (value: string) => value === 'draft_model'
    },
    control: {
      kind: 'choice',
      name: 'draft_selection_policy',
      value: 'auto',
      presentation: 'toggle',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'manual_only', label: 'Manual only' }
      ]
    }
  },
  {
    id: 'mirostat-mode',
    categoryId: 'request-defaults',
    icon: 'brain',
    label: 'Mirostat mode',
    description: 'Controls whether Mirostat is active.',
    inheritedLabel: 'Inherited by request defaults',
    control: {
      kind: 'choice',
      name: 'mirostat_mode',
      value: 'disabled',
      presentation: 'segmented',
      options: [
        { value: 'disabled', label: 'disabled' },
        { value: '1', label: '1' },
        { value: '2', label: '2' }
      ]
    }
  },
  {
    id: 'mirostat-entropy',
    categoryId: 'request-defaults',
    icon: 'gauge',
    label: 'Mirostat entropy',
    description: 'Depends on the Mirostat mode.',
    inheritedLabel: 'Only available when Mirostat is enabled',
    dependsOn: {
      settingId: 'mirostat-mode',
      condition: (value: string) => value !== 'disabled'
    },
    control: {
      kind: 'range',
      name: 'mirostat_entropy',
      value: '5',
      min: 0.1,
      max: 10,
      step: 0.1
    }
  }
] satisfies readonly ConfigurationDefaultsSetting[]

const dependencyData = {
  categories: [
    {
      id: 'speculative-decoding',
      label: 'Speculative Decoding',
      summary: 'Speculative defaults.',
      help: 'Speculative defaults'
    },
    {
      id: 'request-defaults',
      label: 'Request Defaults',
      summary: 'Sampling defaults.',
      help: 'Sampling defaults'
    }
  ],
  settings: dependencySettings,
  preview: []
} satisfies ConfigurationDefaultsHarnessData

const defaultValues: ConfigurationDefaultsValues = {}

function renderDefaultsTab(overrides: Partial<Parameters<typeof DefaultsTab>[0]> = {}) {
  return render(
    <DefaultsTab
      data={overrides.data ?? defaultsData}
      values={overrides.values ?? defaultValues}
      onSettingValueChange={overrides.onSettingValueChange ?? vi.fn()}
      onResetAll={overrides.onResetAll ?? vi.fn()}
      configFilePath={overrides.configFilePath}
    />
  )
}

function previewSource() {
  const source = screen.getByRole('textbox', { name: /\[defaults\] preview code/i })

  if (!(source instanceof HTMLTextAreaElement)) throw new Error('Expected TOML preview textarea')

  return source
}

function defaultsRail() {
  return within(screen.getByRole('navigation', { name: /defaults sections/i }))
}

function settingsRow(label: string) {
  const row = screen.getByText(label).closest('[data-settings-row="true"]')

  if (!(row instanceof HTMLElement)) throw new Error(`Expected settings row for ${label}`)

  return row
}

describe('DefaultsTab', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('hides advanced settings by default and persists the toggle', async () => {
    const user = userEvent.setup()

    renderDefaultsTab({
      values: {
        'advanced-reasoning': '256'
      }
    })

    expect(screen.getByRole('button', { name: /show advanced/i })).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Reasoning' })).not.toBeInTheDocument()
    expect(previewSource().value).not.toContain('runtime_mode = "auto"')
    expect(previewSource().value).toContain('reasoning_budget = 256')

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /hide advanced/i })).toHaveAttribute('aria-pressed', 'true')
      expect(screen.getByRole('heading', { name: 'Reasoning' })).toBeInTheDocument()
      expect(previewSource().value).toContain('[defaults.runtime]')
      expect(previewSource().value).toContain('reasoning_budget = 256')
      expect(window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY)).toBe('true')
    })

    await user.click(screen.getByRole('button', { name: /hide advanced/i }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /show advanced/i })).toHaveAttribute('aria-pressed', 'false')
      expect(screen.queryByRole('heading', { name: 'Reasoning' })).not.toBeInTheDocument()
      expect(previewSource().value).toContain('[defaults.runtime]')
      expect(previewSource().value).toContain('reasoning_budget = 256')
      expect(window.localStorage.getItem(SHOW_ADVANCED_STORAGE_KEY)).toBeNull()
    })
  })

  it('hydrates show advanced from localStorage', () => {
    window.localStorage.setItem(SHOW_ADVANCED_STORAGE_KEY, 'true')

    renderDefaultsTab()

    expect(screen.getByRole('button', { name: /hide advanced/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('heading', { name: 'Reasoning' })).toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('prefill_chunk_schedule')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })

  it('renders the real defaults inventory with integrated categories and section previews', async () => {
    const user = userEvent.setup()

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        threads: '12',
        temperature: '0.8',
        'top-k': '55',
        'activation-wire-dtype': 'q8',
        'binary-stage-transport': 'on',
        'image-min-tokens': '64',
        'mmproj-offload': 'on',
        'server-alias': 'carrack-mesh',
        'direct-io': 'off',
        mlock: 'on'
      }
    })

    const rail = defaultsRail()
    expect(rail.getAllByRole('button')).toHaveLength(6)
    expect(rail.getByRole('button', { name: /runtime/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /request defaults/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /skippy transport/i })).toBeInTheDocument()
    expect(rail.getByRole('button', { name: /multimodal/i })).toBeInTheDocument()
    expect(rail.queryByRole('button', { name: /advanced server/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(screen.queryByText('Memory lock')).not.toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /request defaults/i }))
    expect(rail.getByRole('button', { name: /request defaults/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Request Defaults' })).toBeInTheDocument()
    expect(screen.getByText('Temperature')).toBeInTheDocument()
    expect(screen.getByText('Top-k')).toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /skippy transport/i }))
    expect(rail.getByRole('button', { name: /skippy transport/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Skippy Transport' })).toBeInTheDocument()
    expect(screen.getByText('Activation wire dtype')).toBeInTheDocument()

    await user.click(rail.getByRole('button', { name: /multimodal/i }))
    expect(rail.getByRole('button', { name: /multimodal/i })).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('heading', { name: 'Multimodal' })).toBeInTheDocument()
    expect(screen.getByText('MMProj offload')).toBeInTheDocument()

    expect(screen.getByRole('slider', { name: 'CPU threads' })).toHaveValue('12')
    expect(screen.getByRole('slider', { name: 'Temperature' })).toHaveValue('0.8')
    expect(screen.getByRole('slider', { name: 'Top-k' })).toHaveValue('55')
    expect(previewSource().value).toContain('threads = 12')
    expect(previewSource().value).toContain('[defaults.request_defaults]')
    expect(previewSource().value).toContain('temperature = 0.8')
    expect(previewSource().value).toContain('top_k = 55')
    expect(previewSource().value).toContain('[defaults.skippy]')
    expect(previewSource().value).toContain('activation_wire_dtype = "q8"')
    expect(previewSource().value).toContain('binary_stage_transport = "on"')
    expect(previewSource().value).toContain('mlock = true')
    expect(previewSource().value).toContain('[defaults.multimodal]')
    expect(previewSource().value).toContain('image_min_tokens = 64')
    expect(previewSource().value).toContain('mmproj_offload = "on"')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          threads: '12',
          temperature: '0.8',
          'top-k': '55',
          'activation-wire-dtype': 'q8',
          'binary-stage-transport': 'on',
          'image-min-tokens': '64',
          'mmproj-offload': 'on',
          'direct-io': 'off',
          mlock: 'on'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(previewSource().value).not.toContain('[defaults.advanced.server]')
  })

  it('omits default-only metadata sections while keeping hidden advanced non-default values in preview', () => {
    renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        mlock: 'on',
        'server-alias': 'carrack-mesh'
      }
    })

    expect(screen.queryByText('Memory lock')).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('[defaults.hardware]')
    expect(previewSource().value).toContain('mlock = true')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')
    expect(previewSource().value).not.toContain('[defaults.skippy]')
    expect(previewSource().value).not.toContain('[defaults.multimodal]')
  })

  it('uses canonical inventory defaults when previewing hydrated live settings', () => {
    const liveHydratedDefaults = {
      ...CONFIGURATION_DEFAULTS,
      settings: CONFIGURATION_DEFAULTS.settings.map((setting) => {
        if (setting.id === 'activation-wire-dtype') {
          return {
            ...setting,
            control: {
              ...setting.control,
              value: 'q8'
            }
          }
        }

        if (setting.id === 'image-min-tokens') {
          return {
            ...setting,
            control: {
              ...setting.control,
              value: '64'
            }
          }
        }

        if (setting.id === 'server-alias') {
          return {
            ...setting,
            control: {
              ...setting.control,
              value: 'carrack-mesh'
            }
          }
        }

        return setting
      })
    } satisfies ConfigurationDefaultsHarnessData

    renderDefaultsTab({
      data: liveHydratedDefaults,
      values: {}
    })

    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('[defaults.skippy]')
    expect(previewSource().value).toContain('activation_wire_dtype = "q8"')
    expect(previewSource().value).toContain('[defaults.multimodal]')
    expect(previewSource().value).toContain('image_min_tokens = 64')
    expect(previewSource().value).toContain('[defaults.advanced.server]')
    expect(previewSource().value).toContain('alias = "carrack-mesh"')
    expect(previewSource().value).not.toContain('[defaults.request_defaults]')
  })

  it('shows restart badges only for restart-required settings and exposes the exact tooltip copy on hover and focus', async () => {
    const user = userEvent.setup()

    const { unmount } = renderDefaultsTab({ data: CONFIGURATION_DEFAULTS })

    const cpuThreadsRow = within(settingsRow('CPU threads'))
    const topKRow = within(settingsRow('Top-k'))
    const restartBadge = cpuThreadsRow.getByRole('button', { name: 'Restart required' })

    expect(restartBadge).toBeInTheDocument()
    expect(topKRow.queryByRole('button', { name: 'Restart required' })).not.toBeInTheDocument()

    await user.hover(restartBadge)
    expect(await screen.findByText(RESTART_REQUIRED_TOOLTIP, { selector: 'div' })).toBeInTheDocument()

    unmount()
    renderDefaultsTab({ data: CONFIGURATION_DEFAULTS })

    within(settingsRow('CPU threads')).getByRole('button', { name: 'Restart required' }).focus()
    expect(await screen.findByText(RESTART_REQUIRED_TOOLTIP, { selector: 'div' })).toBeInTheDocument()
  })

  it('keeps advanced filtering consistent across real category counts, rows, and section visibility', async () => {
    const user = userEvent.setup()

    renderDefaultsTab({ data: CONFIGURATION_DEFAULTS })

    const rail = defaultsRail()
    expect(rail.getAllByRole('button')).toHaveLength(6)
    expect(screen.queryByText('Mirostat mode')).not.toBeInTheDocument()
    expect(screen.queryByText('Server alias')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show advanced/i }))

    await waitFor(() => {
      expect(rail.getAllByRole('button')).toHaveLength(7)
      expect(rail.getByRole('button', { name: /advanced server/i })).toBeInTheDocument()
      expect(screen.getByText('Mirostat mode')).toBeInTheDocument()
      expect(screen.getByText('Server alias')).toBeInTheDocument()
    })

    await user.click(rail.getByRole('button', { name: /advanced server/i }))
    expect(rail.getByRole('button', { name: /advanced server/i })).toHaveAttribute('aria-current', 'true')

    await user.click(screen.getByRole('button', { name: /hide advanced/i }))

    await waitFor(() => {
      expect(rail.getAllByRole('button')).toHaveLength(6)
      expect(rail.queryByRole('button', { name: /advanced server/i })).not.toBeInTheDocument()
      expect(screen.queryByText('Mirostat mode')).not.toBeInTheDocument()
      expect(screen.queryByText('Server alias')).not.toBeInTheDocument()
      expect(rail.getByRole('button', { name: /runtime/i })).toHaveAttribute('aria-current', 'true')
    })
  })

  it('keeps integrated dependency disable states and explanations working across real dependency pairs', () => {
    window.localStorage.setItem(SHOW_ADVANCED_STORAGE_KEY, 'true')

    const { rerender } = renderDefaultsTab({
      data: CONFIGURATION_DEFAULTS,
      values: {
        'speculation-mode': 'ngram'
      }
    })

    expect(
      screen.getByText('Default draft selection policy').closest('[data-settings-row-disabled="true"]')
    ).not.toBeNull()
    expect(screen.getAllByText('Requires speculation-mode = draft')).toHaveLength(5)
    expect(screen.getByText('Prefill chunk size').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Requires prefill-chunking = fixed')).toBeInTheDocument()
    expect(screen.getByText('Prefill chunk schedule').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Requires prefill-chunking = schedule')).toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Mirostat learning rate').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getAllByText('Requires mirostat-mode = 1 or 2')).toHaveLength(2)
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('prefill_chunk_schedule')
    expect(previewSource().value).not.toContain('mirostat_entropy')

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          'speculation-mode': 'draft',
          'mirostat-mode': '2',
          'prefill-chunking': 'fixed'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(screen.getByText('Default draft selection policy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryAllByText('Requires speculation-mode = draft')).toHaveLength(0)
    expect(screen.getByText('Prefill chunk size').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = fixed')).not.toBeInTheDocument()
    expect(screen.getByText('Prefill chunk schedule').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Requires prefill-chunking = schedule')).toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.getByText('Mirostat learning rate').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires mirostat-mode = 1 or 2')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('mirostat_mode = 2')
    expect(previewSource().value).toContain('prefill_chunking = "fixed"')
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('mirostat_entropy')

    rerender(
      <DefaultsTab
        data={CONFIGURATION_DEFAULTS}
        values={{
          'speculation-mode': 'draft',
          'mirostat-mode': '2',
          'prefill-chunking': 'schedule',
          'prefill-chunk-schedule': '128,256'
        }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(screen.getByText('Prefill chunk size').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Requires prefill-chunking = fixed')).toBeInTheDocument()
    expect(screen.getByText('Prefill chunk schedule').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires prefill-chunking = schedule')).not.toBeInTheDocument()
    expect(previewSource().value).toContain('prefill_chunk_schedule = "128,256"')
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('prefill_chunk_size')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })

  it('disables dependent settings until their dependency is satisfied', () => {
    const { rerender } = renderDefaultsTab({ data: dependencyData })

    expect(screen.getByText('Draft selection policy').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Requires speculation-mode = draft_model')).toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).not.toBeNull()
    expect(screen.getByText('Requires mirostat-mode = 1 or 2')).toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('mirostat_entropy')

    rerender(
      <DefaultsTab
        data={dependencyData}
        values={{ 'speculation-mode': 'draft_model', 'mirostat-mode': '2' }}
        onSettingValueChange={vi.fn()}
        onResetAll={vi.fn()}
      />
    )

    expect(screen.getByText('Draft selection policy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires speculation-mode = draft_model')).not.toBeInTheDocument()
    expect(screen.getByText('Mirostat entropy').closest('[data-settings-row-disabled="true"]')).toBeNull()
    expect(screen.queryByText('Requires mirostat-mode = 1 or 2')).not.toBeInTheDocument()
    expect(previewSource().value).not.toContain('draft_selection_policy')
    expect(previewSource().value).not.toContain('mirostat_entropy')
  })
})
