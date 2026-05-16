import { describe, expect, it } from 'vitest'

import { CONFIGURATION_DEFAULTS } from '@/features/app-tabs/data'
import type {
  ConfigurationDefaultsCategory,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsSetting
} from '@/features/app-tabs/types'

const EXPECTED_SETTING_IDS = [
  'threads',
  'threads-batch',
  'continuous-batching',
  'numa',
  'cpu-affinity',
  'priority',
  'poll',
  'slot-prompt-similarity',
  'gpu-layers',
  'mmap',
  'mlock',
  'warmup',
  'direct-io',
  'split-mode',
  'main-gpu',
  'tensor-split'
] as const

const MEMORY_SETTING_IDS = [
  'kv-cache',
  'memory-margin',
  'ctx-size',
  'batch',
  'ubatch',
  'cache-type-k',
  'cache-type-v',
  'kv-offload',
  'kv-unified',
  'cache-ram-mib',
  'cache-idle-slots',
  'prompt-cache',
  'context-shift'
] as const

const ADVANCED_IDS = new Set([
  'numa',
  'cpu-affinity',
  'priority',
  'poll',
  'slot-prompt-similarity',
  'mmap',
  'mlock',
  'warmup',
  'direct-io',
  'split-mode',
  'main-gpu',
  'tensor-split'
])

const MEMORY_ADVANCED_IDS = new Set([
  'ubatch',
  'kv-offload',
  'kv-unified',
  'cache-ram-mib',
  'cache-idle-slots',
  'prompt-cache',
  'context-shift'
])

const MEMORY_LEGACY_IDS = new Set(['kv-cache', 'memory-margin'])

const REQUEST_DEFAULT_SETTING_IDS = [
  'temperature',
  'top-p',
  'top-k',
  'min-p',
  'presence-penalty',
  'frequency-penalty',
  'max-tokens',
  'seed',
  'ignore-eos',
  'mirostat-mode',
  'mirostat-entropy',
  'mirostat-learning-rate',
  'samplers',
  'sampler-sequence',
  'stop'
] as const

const REQUEST_DEFAULTS_ADVANCED_IDS = new Set([
  'seed',
  'ignore-eos',
  'mirostat-mode',
  'mirostat-entropy',
  'mirostat-learning-rate',
  'samplers',
  'sampler-sequence',
  'stop'
])

const TRANSPORT_SETTING_IDS = [
  'activation-wire-dtype',
  'stage-model-path',
  'stage-role',
  'stage-topology',
  'prefill-chunking',
  'prefill-chunk-size',
  'prefill-chunk-schedule',
  'binary-stage-transport',
  'lifecycle-startup-timeout-ms',
  'lifecycle-readiness-interval-ms',
  'lifecycle-health-interval-ms'
] as const

const TRANSPORT_ADVANCED_IDS = new Set([
  'stage-model-path',
  'stage-role',
  'stage-topology',
  'prefill-chunk-schedule',
  'lifecycle-startup-timeout-ms',
  'lifecycle-readiness-interval-ms',
  'lifecycle-health-interval-ms'
])

const MULTIMODAL_SETTING_IDS = [
  'mmproj-offload',
  'image-min-tokens',
  'image-max-tokens',
  'mmproj',
  'mmproj-url'
] as const

const ADVANCED_SERVER_SETTING_IDS = ['server-alias'] as const

const RUNTIME_THROUGHPUT_IDS = new Set([
  'threads',
  'threads-batch',
  'continuous-batching',
  'numa',
  'cpu-affinity',
  'priority',
  'poll',
  'slot-prompt-similarity'
])

const RUNTIME_HARDWARE_IDS = new Set([
  'gpu-layers',
  'mmap',
  'mlock',
  'warmup',
  'direct-io',
  'split-mode',
  'main-gpu',
  'tensor-split'
])

function settingById(id: string): ConfigurationDefaultsSetting {
  const setting = CONFIGURATION_DEFAULTS.settings.find((entry: ConfigurationDefaultsSetting) => entry.id === id)

  if (!setting) {
    throw new Error(`Expected setting ${id}`)
  }

  return setting
}

function categoryById(id: string): ConfigurationDefaultsCategory {
  const category = CONFIGURATION_DEFAULTS.categories.find((entry: ConfigurationDefaultsCategory) => entry.id === id)

  if (!category) {
    throw new Error(`Expected category ${id}`)
  }

  return category
}

function runtimeHardwareSettings(): readonly ConfigurationDefaultsSetting[] {
  return CONFIGURATION_DEFAULTS.settings.slice(0, EXPECTED_SETTING_IDS.length) as readonly ConfigurationDefaultsSetting[]
}

function memorySettings(): readonly ConfigurationDefaultsSetting[] {
  const start = CONFIGURATION_DEFAULTS.settings.findIndex((setting: ConfigurationDefaultsSetting) => setting.id === 'kv-cache')

  if (start < 0) {
    throw new Error('Expected kv-cache setting')
  }

  return CONFIGURATION_DEFAULTS.settings.slice(
    start,
    start + MEMORY_SETTING_IDS.length
  ) as readonly ConfigurationDefaultsSetting[]
}

function transportSettings(): readonly ConfigurationDefaultsSetting[] {
  const start = CONFIGURATION_DEFAULTS.settings.findIndex((setting: ConfigurationDefaultsSetting) => setting.id === 'activation-wire-dtype')

  if (start < 0) {
    throw new Error('Expected activation-wire-dtype setting')
  }

  return CONFIGURATION_DEFAULTS.settings.slice(
    start,
    start + TRANSPORT_SETTING_IDS.length
  ) as readonly ConfigurationDefaultsSetting[]
}

function requestDefaultSettings(): readonly ConfigurationDefaultsSetting[] {
  const start = CONFIGURATION_DEFAULTS.settings.findIndex((setting: ConfigurationDefaultsSetting) => setting.id === 'temperature')

  if (start < 0) {
    throw new Error('Expected temperature setting')
  }

  return CONFIGURATION_DEFAULTS.settings.slice(
    start,
    start + REQUEST_DEFAULT_SETTING_IDS.length
  ) as readonly ConfigurationDefaultsSetting[]
}

function expectChoice(id: string, name: string, value: string, presentation: 'segmented' | 'select' | 'toggle') {
  const setting = settingById(id)
  const control = setting.control as Extract<ConfigurationDefaultsControl, { kind: 'choice' }>

  expect(control.kind).toBe('choice')
  expect(control.name).toBe(name)
  expect(control.value).toBe(value)
  expect(control.presentation).toBe(presentation)
}

describe('CONFIGURATION_DEFAULTS runtime and hardware inventory', () => {
  it('declares the full defaults inventory and multimodal/server additions', () => {
    expect(CONFIGURATION_DEFAULTS.categories).toEqual([
      {
        id: 'runtime',
        label: 'Runtime',
        summary: 'Throughput, scheduling, and hardware defaults.',
        help: 'Default throughput and hardware policy'
      },
      {
        id: 'memory',
        label: 'Memory',
        summary: 'KV cache policy and fit headroom.',
        help: 'VRAM accounting and KV cache policy'
      },
      {
        id: 'speculative-decoding',
        label: 'Speculative Decoding',
        summary: 'Draft acceleration defaults.',
        help: 'Speculative draft model and acceptance defaults',
        tomlSection: 'defaults.speculative'
      },
      {
        id: 'advanced',
        label: 'Reasoning',
        summary: 'Reasoning and repetition controls.',
        help: 'Reasoning and sampling defaults'
      },
      {
        id: 'request-defaults',
        label: 'Request Defaults',
        summary: 'Sampling and response defaults.',
        help: 'Sampling, stopping, and repeat-penalty defaults'
      },
      {
        id: 'skippy-transport',
        label: 'Skippy Transport',
        summary: 'Activation wire dtype, prefill chunking, and lifecycle timing.',
        help: 'Stage transport, chunking, and lifecycle defaults',
        tomlSection: 'defaults.skippy'
      },
      {
        id: 'multimodal',
        label: 'Multimodal',
        summary: 'Projector and image token defaults.',
        help: 'Vision projector and image token defaults',
        tomlSection: 'defaults.multimodal'
      },
      {
        id: 'advanced-server',
        label: 'Advanced Server',
        summary: 'Server identity and operator overrides.',
        help: 'Advanced server defaults and identity overrides',
        tomlSection: 'defaults.advanced.server'
      }
    ])
    expect(CONFIGURATION_DEFAULTS.settings).toHaveLength(75)
    expect(CONFIGURATION_DEFAULTS.settings.slice(0, EXPECTED_SETTING_IDS.length).map((setting: ConfigurationDefaultsSetting) => setting.id)).toEqual(
      EXPECTED_SETTING_IDS
    )

    for (const setting of runtimeHardwareSettings()) {
      const tomlSection = (setting as ConfigurationDefaultsSetting & { tomlSection?: string }).tomlSection

      expect(setting.categoryId).toBe('runtime')
      expect(setting.mutability).toBe('restart-required')
      if (RUNTIME_THROUGHPUT_IDS.has(setting.id)) {
        expect(tomlSection).toBe('defaults.throughput')
      } else if (RUNTIME_HARDWARE_IDS.has(setting.id)) {
        expect(tomlSection).toBe('defaults.hardware')
      } else {
        expect(tomlSection).toBeUndefined()
      }

      if (ADVANCED_IDS.has(setting.id)) {
        expect(setting.visibility).toBe('advanced')
      } else {
        expect(setting.visibility).toBeUndefined()
      }
    }

    const threads = settingById('threads')
    expect(threads.control).toMatchObject({ kind: 'range', name: 'threads', value: '0', min: 0, max: 256, step: 1, unit: 'threads' })

    const threadsBatch = settingById('threads-batch')
    expect(threadsBatch.control).toMatchObject({
      kind: 'range',
      name: 'threads_batch',
      value: '0',
      min: 0,
      max: 256,
      step: 1,
      unit: 'threads'
    })

    expectChoice('continuous-batching', 'continuous_batching', 'auto', 'segmented')
    expect(settingById('continuous-batching').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expectChoice('numa', 'numa', 'auto', 'select')
    expect(settingById('numa').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'disabled', label: 'disabled' },
        { value: 'distribute', label: 'distribute' },
        { value: 'isolate', label: 'isolate' },
        { value: 'numactl', label: 'numactl' }
      ]
    })

    expect(settingById('cpu-affinity').control).toMatchObject({ kind: 'text', name: 'cpu_affinity', placeholder: 'e.g. 0-3,8-11' })
    expect(settingById('priority').control).toMatchObject({ kind: 'text', name: 'priority', placeholder: 'e.g. 0 or normal' })

    expectChoice('poll', 'poll', 'auto', 'segmented')
    expect(settingById('poll').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'busy', label: 'busy' },
        { value: 'sleep', label: 'sleep' }
      ]
    })

    expect(settingById('slot-prompt-similarity').control).toMatchObject({
      kind: 'range',
      name: 'slot_prompt_similarity',
      value: '0.50',
      min: 0,
      max: 1,
      step: 0.01
    })

    const gpuLayers = settingById('gpu-layers')
    expect(gpuLayers.control).toMatchObject({ kind: 'text', name: 'gpu_layers', value: 'auto', placeholder: 'auto or integer layer count' })
    expect(gpuLayers.description).toContain('-1')

    expectChoice('mmap', 'mmap', 'auto', 'segmented')
    expect(settingById('mmap').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expectChoice('mlock', 'mlock', 'off', 'toggle')
    expect(settingById('mlock').control).toMatchObject({ options: [{ value: 'on', label: 'on' }, { value: 'off', label: 'off' }] })

    expectChoice('warmup', 'warmup', 'auto', 'segmented')
    expect(settingById('warmup').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expectChoice('direct-io', 'direct_io', 'off', 'toggle')
    expect(settingById('direct-io').control).toMatchObject({ options: [{ value: 'on', label: 'on' }, { value: 'off', label: 'off' }] })

    expectChoice('split-mode', 'split_mode', 'auto', 'select')
    expect(settingById('split-mode').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'none', label: 'none' },
        { value: 'layer', label: 'layer' },
        { value: 'row', label: 'row' }
      ]
    })

    expect(settingById('main-gpu').control).toMatchObject({ kind: 'range', name: 'main_gpu', value: '0', min: 0, max: 7, step: 1, unit: 'GPU index' })
    expect(settingById('tensor-split').control).toMatchObject({ kind: 'text', name: 'tensor_split', placeholder: 'e.g. 0.5,0.5' })

    expect(memorySettings().map((setting) => setting.id)).toEqual(MEMORY_SETTING_IDS)
    expect(new Set(memorySettings().map((setting) => setting.id)).size).toBe(MEMORY_SETTING_IDS.length)

    for (const setting of memorySettings()) {
      expect(setting.categoryId).toBe('memory')
      if (MEMORY_LEGACY_IDS.has(setting.id)) {
        expect(setting.mutability).toBeUndefined()
        expect(setting.visibility).toBeUndefined()
      } else {
        expect(setting.mutability).toBe('restart-required')
        expect((setting as ConfigurationDefaultsSetting & { tomlSection?: string }).tomlSection).toBe('defaults.model_fit')
        if (MEMORY_ADVANCED_IDS.has(setting.id)) {
          expect(setting.visibility).toBe('advanced')
        } else {
          expect(setting.visibility).toBeUndefined()
        }
      }
    }

    expect(settingById('kv-cache')).toMatchObject({ categoryId: 'memory' })
    expect(settingById('kv-cache').control).toMatchObject({
      kind: 'choice',
      name: 'kv_cache_policy',
      value: 'auto',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'quality', label: 'quality' },
        { value: 'balanced', label: 'balanced' },
        { value: 'saver', label: 'saver' }
      ]
    })

    expect(settingById('memory-margin').control).toMatchObject({
      kind: 'range',
      name: 'safety_margin_gb',
      value: '2',
      min: 0,
      max: 8,
      step: 0.5,
      unit: 'GB'
    })

    expect(settingById('ctx-size').control).toMatchObject({
      kind: 'range',
      name: 'ctx_size',
      value: '512',
      min: 512,
      max: 131072,
      step: 512,
      unit: 'tokens'
    })

    expect(settingById('batch').control).toMatchObject({
      kind: 'range',
      name: 'batch',
      value: '512',
      min: 32,
      max: 4096,
      step: 32,
      unit: 'tokens'
    })

    expect(settingById('ubatch').control).toMatchObject({
      kind: 'range',
      name: 'ubatch',
      value: '512',
      min: 32,
      max: 4096,
      step: 32,
      unit: 'tokens'
    })

    expect(settingById('cache-type-k').control).toMatchObject({
      kind: 'choice',
      name: 'cache_type_k',
      value: 'f16',
      presentation: 'segmented',
      options: [
        { value: 'f16', label: 'f16' },
        { value: 'q8_0', label: 'q8_0' },
        { value: 'q4_0', label: 'q4_0' }
      ]
    })

    expect(settingById('cache-type-v').control).toMatchObject({
      kind: 'choice',
      name: 'cache_type_v',
      value: 'f16',
      presentation: 'segmented',
      options: [
        { value: 'f16', label: 'f16' },
        { value: 'q8_0', label: 'q8_0' },
        { value: 'q4_0', label: 'q4_0' }
      ]
    })

    expect(settingById('kv-offload').control).toMatchObject({
      kind: 'choice',
      name: 'kv_offload',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expect(settingById('kv-unified').control).toMatchObject({
      kind: 'choice',
      name: 'kv_unified',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expect(settingById('cache-ram-mib').control).toMatchObject({
      kind: 'range',
      name: 'cache_ram_mib',
      value: '8192',
      min: 0,
      max: 65536,
      step: 256,
      unit: 'MiB'
    })

    expect(settingById('cache-idle-slots').control).toMatchObject({
      kind: 'range',
      name: 'cache_idle_slots',
      value: '4',
      min: 0,
      max: 64,
      step: 1,
      unit: 'slots'
    })

    expect(settingById('prompt-cache').control).toMatchObject({
      kind: 'choice',
      name: 'prompt_cache',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expect(settingById('context-shift').control).toMatchObject({
      kind: 'choice',
      name: 'context_shift',
      value: 'auto',
      presentation: 'segmented',
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expect(settingById('draft-min-tokens').control).toMatchObject({
      kind: 'range',
      name: 'draft_min_tokens',
      value: '0',
      min: 0,
      max: 32,
      step: 1,
      unit: 'tokens'
    })
    expect(settingById('draft-min-tokens').mutability).toBe('restart-required')

    expect(settingById('draft-acceptance-threshold').visibility).toBe('advanced')
    expect(settingById('draft-acceptance-threshold').mutability).toBe('restart-required')
    expect(settingById('draft-acceptance-threshold').control).toMatchObject({
      kind: 'range',
      name: 'draft_acceptance_threshold',
      value: '0.70',
      min: 0,
      max: 1,
      step: 0.05
    })

    expect(requestDefaultSettings().map((setting: ConfigurationDefaultsSetting) => setting.id)).toEqual(REQUEST_DEFAULT_SETTING_IDS)
    expect(new Set(requestDefaultSettings().map((setting: ConfigurationDefaultsSetting) => setting.id)).size).toBe(REQUEST_DEFAULT_SETTING_IDS.length)

    for (const setting of requestDefaultSettings()) {
      const tomlSection = (setting as ConfigurationDefaultsSetting & { tomlSection?: string }).tomlSection

      expect(setting.categoryId).toBe('request-defaults')
      expect(setting.mutability).toBe('runtime')
      expect(tomlSection).toBe('defaults.request_defaults')

      if (REQUEST_DEFAULTS_ADVANCED_IDS.has(setting.id)) {
        expect(setting.visibility).toBe('advanced')
      } else {
        expect(setting.visibility).toBeUndefined()
      }
    }

    expect(settingById('temperature').control).toMatchObject({ kind: 'range', name: 'temperature', value: '1.0', min: 0, max: 2, step: 0.05 })
    expect(settingById('top-p').control).toMatchObject({ kind: 'range', name: 'top_p', value: '1.0', min: 0, max: 1, step: 0.05 })
    expect(settingById('top-k').control).toMatchObject({ kind: 'range', name: 'top_k', value: '40', min: 0, max: 100, step: 1 })
    expect(settingById('min-p').control).toMatchObject({ kind: 'range', name: 'min_p', value: '0.05', min: 0, max: 1, step: 0.05 })
    expect(settingById('presence-penalty').control).toMatchObject({ kind: 'range', name: 'presence_penalty', value: '0', min: 0, max: 2, step: 0.1 })
    expect(settingById('frequency-penalty').control).toMatchObject({ kind: 'range', name: 'frequency_penalty', value: '0', min: 0, max: 2, step: 0.1 })
    expect(settingById('max-tokens').control).toMatchObject({ kind: 'range', name: 'max_tokens', value: '0', min: 0, max: 32768, step: 256, unit: 'tokens' })
    expect(settingById('seed').control).toMatchObject({ kind: 'text', name: 'seed', value: '-1', placeholder: '-1 (random)' })
    expect(settingById('ignore-eos').control).toMatchObject({
      kind: 'choice',
      name: 'ignore_eos',
      value: 'off',
      presentation: 'toggle',
      options: [
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expectChoice('mirostat-mode', 'mirostat_mode', 'disabled', 'segmented')
    expect(settingById('mirostat-mode').control).toMatchObject({
      options: [
        { value: 'disabled', label: 'disabled' },
        { value: '1', label: '1' },
        { value: '2', label: '2' }
      ]
    })

    expect(settingById('mirostat-entropy').control).toMatchObject({ kind: 'range', name: 'mirostat_entropy', value: '5', min: 0.1, max: 10, step: 0.1 })
    expect(settingById('mirostat-entropy').dependsOn?.settingId).toBe('mirostat-mode')
    expect(settingById('mirostat-entropy').dependsOn?.condition('disabled')).toBe(false)
    expect(settingById('mirostat-entropy').dependsOn?.condition('1')).toBe(true)

    expect(settingById('mirostat-learning-rate').control).toMatchObject({
      kind: 'range',
      name: 'mirostat_learning_rate',
      value: '0.1',
      min: 0.01,
      max: 1,
      step: 0.01
    })
    expect(settingById('mirostat-learning-rate').dependsOn?.settingId).toBe('mirostat-mode')
    expect(settingById('mirostat-learning-rate').dependsOn?.condition('disabled')).toBe(false)
    expect(settingById('mirostat-learning-rate').dependsOn?.condition('2')).toBe(true)

    expect(settingById('samplers').control).toMatchObject({
      kind: 'text',
      name: 'samplers',
      value: '',
      placeholder: 'top_k,tfs_z,typical_p,top_p,min_p,temperature'
    })

    expect(settingById('sampler-sequence').control).toMatchObject({
      kind: 'text',
      name: 'sampler_sequence',
      value: '',
      placeholder: 'e.g. top_k;top_p;temperature'
    })

    expect(settingById('stop').control).toMatchObject({
      kind: 'text',
      name: 'stop',
      value: '',
      placeholder: 'comma-separated stop sequences'
    })

    expect(settingById('reasoning-format').categoryId).toBe('advanced')
    expect(settingById('reasoning-format').control).toMatchObject({
      kind: 'choice',
      name: 'reasoning_format',
      value: 'deepseek',
      options: [
        { value: 'deepseek', label: 'deepseek' },
        { value: 'qwen', label: 'qwen' },
        { value: 'off', label: 'off' }
      ]
    })
    expect(settingById('reasoning-budget').control).toMatchObject({ kind: 'range', name: 'reasoning_budget', value: '0', min: 0, max: 4096, step: 128, unit: 'tok' })
    expect(settingById('repeat-penalty').control).toMatchObject({ kind: 'range', name: 'repeat_penalty', value: '1.1', min: 1, max: 2, step: 0.05 })
    expect(settingById('repeat-last-n').control).toMatchObject({ kind: 'range', name: 'repeat_last_n', value: '256', min: 0, max: 1024, step: 32, unit: 'tok' })

    const skippyTransportCategory = categoryById('skippy-transport') as ConfigurationDefaultsCategory & {
      tomlSection?: string
    }
    expect(skippyTransportCategory).toMatchObject({
      label: 'Skippy Transport',
      summary: 'Activation wire dtype, prefill chunking, and lifecycle timing.',
      help: 'Stage transport, chunking, and lifecycle defaults',
      tomlSection: 'defaults.skippy'
    })

    expect(transportSettings().map((setting) => setting.id)).toEqual(TRANSPORT_SETTING_IDS)
    expect(new Set(transportSettings().map((setting) => setting.id)).size).toBe(TRANSPORT_SETTING_IDS.length)

    for (const setting of transportSettings()) {
      const tomlSection = (setting as ConfigurationDefaultsSetting & { tomlSection?: string }).tomlSection

      expect(setting.categoryId).toBe('skippy-transport')
      expect(setting.mutability).toBe('restart-required')
      expect(tomlSection).toBe('defaults.skippy')

      if (TRANSPORT_ADVANCED_IDS.has(setting.id)) {
        expect(setting.visibility).toBe('advanced')
      } else {
        expect(setting.visibility).toBeUndefined()
      }
    }

    expect(settingById('activation-wire-dtype').icon).toBe('zap')
    expect(settingById('stage-model-path').icon).toBe('folder')
    expect(settingById('stage-role').icon).toBe('server')
    expect(settingById('stage-topology').icon).toBe('layers')
    expect(settingById('prefill-chunking').icon).toBe('layers')
    expect(settingById('prefill-chunk-size').icon).toBe('gauge')
    expect(settingById('prefill-chunk-schedule').icon).toBe('layers')
    expect(settingById('binary-stage-transport').icon).toBe('binary')
    expect(settingById('lifecycle-startup-timeout-ms').icon).toBe('gauge')
    expect(settingById('lifecycle-readiness-interval-ms').icon).toBe('gauge')
    expect(settingById('lifecycle-health-interval-ms').icon).toBe('shield')

    expectChoice('activation-wire-dtype', 'activation_wire_dtype', 'auto', 'segmented')
    expect(settingById('activation-wire-dtype').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'f16', label: 'f16' },
        { value: 'f32', label: 'f32' },
        { value: 'q8', label: 'q8' }
      ]
    })

    expect(settingById('stage-model-path').control).toMatchObject({
      kind: 'text',
      name: 'stage_model_path',
      placeholder: 'e.g. hf://meshllm/... or /path/to/stage.gguf'
    })

    expectChoice('stage-role', 'stage_role', 'auto', 'select')
    expect(settingById('stage-role').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'prompt', label: 'prompt' },
        { value: 'stage', label: 'stage' }
      ]
    })

    expect(settingById('stage-topology').control).toMatchObject({
      kind: 'text',
      name: 'stage_topology',
      placeholder: 'topology name or path'
    })

    expectChoice('prefill-chunking', 'prefill_chunking', 'auto', 'select')
    expect(settingById('prefill-chunking').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'fixed', label: 'fixed' },
        { value: 'schedule', label: 'schedule' },
        { value: 'adaptive-ramp', label: 'adaptive-ramp' }
      ]
    })

    expect(settingById('prefill-chunk-size').control).toMatchObject({
      kind: 'range',
      name: 'prefill_chunk_size',
      value: '0',
      min: 0,
      max: 8192,
      step: 64,
      unit: 'tokens'
    })
    expect(settingById('prefill-chunk-size').dependsOn?.settingId).toBe('prefill-chunking')
    expect(settingById('prefill-chunk-size').dependsOn?.condition('fixed')).toBe(true)
    expect(settingById('prefill-chunk-size').dependsOn?.condition('schedule')).toBe(false)

    expect(settingById('prefill-chunk-schedule').control).toMatchObject({
      kind: 'text',
      name: 'prefill_chunk_schedule',
      placeholder: 'e.g. 512,1024,2048'
    })
    expect(settingById('prefill-chunk-schedule').dependsOn?.settingId).toBe('prefill-chunking')
    expect(settingById('prefill-chunk-schedule').dependsOn?.condition('schedule')).toBe(true)
    expect(settingById('prefill-chunk-schedule').dependsOn?.condition('fixed')).toBe(false)

    expectChoice('binary-stage-transport', 'binary_stage_transport', 'auto', 'segmented')
    expect(settingById('binary-stage-transport').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })

    expect(settingById('lifecycle-startup-timeout-ms').control).toMatchObject({
      kind: 'range',
      name: 'lifecycle_startup_timeout_ms',
      value: '30000',
      min: 1,
      max: 600000,
      step: 1000,
      unit: 'ms'
    })

    expect(settingById('lifecycle-readiness-interval-ms').control).toMatchObject({
      kind: 'range',
      name: 'lifecycle_readiness_interval_ms',
      value: '1000',
      min: 100,
      max: 60000,
      step: 100,
      unit: 'ms'
    })

    expect(settingById('lifecycle-health-interval-ms').control).toMatchObject({
      kind: 'range',
      name: 'lifecycle_health_interval_ms',
      value: '15000',
      min: 100,
      max: 60000,
      step: 100,
      unit: 'ms'
    })

    const multimodalSettings = CONFIGURATION_DEFAULTS.settings.filter(
      (setting: ConfigurationDefaultsSetting) => setting.categoryId === 'multimodal'
    ) as readonly (ConfigurationDefaultsSetting & {
      tomlSection?: string
      visibility?: 'standard' | 'advanced'
      mutability?: 'runtime' | 'restart-required'
    })[]
    expect(multimodalSettings.map((setting) => setting.id)).toEqual(MULTIMODAL_SETTING_IDS)
    expect(new Set(multimodalSettings.map((setting) => setting.id)).size).toBe(MULTIMODAL_SETTING_IDS.length)

    for (const setting of multimodalSettings) {
      expect(setting.mutability).toBe('restart-required')
      expect(setting.tomlSection).toBe('defaults.multimodal')
      if (setting.id === 'mmproj' || setting.id === 'mmproj-url') {
        expect(setting.visibility).toBe('advanced')
      } else {
        expect(setting.visibility).toBeUndefined()
      }
    }

    expectChoice('mmproj-offload', 'mmproj_offload', 'auto', 'segmented')
    expect(settingById('mmproj-offload').control).toMatchObject({
      options: [
        { value: 'auto', label: 'auto' },
        { value: 'on', label: 'on' },
        { value: 'off', label: 'off' }
      ]
    })
    expect(settingById('image-min-tokens').control).toMatchObject({
      kind: 'range',
      name: 'image_min_tokens',
      value: '0',
      min: 0,
      max: 2048,
      step: 32,
      unit: 'tokens'
    })
    expect(settingById('image-max-tokens').control).toMatchObject({
      kind: 'range',
      name: 'image_max_tokens',
      value: '2048',
      min: 0,
      max: 4096,
      step: 32,
      unit: 'tokens'
    })
    expect(settingById('mmproj').control).toMatchObject({
      kind: 'text',
      name: 'mmproj',
      value: '',
      placeholder: 'e.g. /path/to/mmproj.gguf'
    })
    expect(settingById('mmproj-url').control).toMatchObject({
      kind: 'text',
      name: 'mmproj_url',
      value: '',
      placeholder: 'e.g. https://example.com/mmproj.gguf'
    })

    expect(settingById('mmproj-offload').tomlSection).toBe('defaults.multimodal')
    expect(settingById('image-min-tokens').tomlSection).toBe('defaults.multimodal')
    expect(settingById('image-max-tokens').tomlSection).toBe('defaults.multimodal')
    expect(settingById('mmproj').tomlSection).toBe('defaults.multimodal')
    expect(settingById('mmproj-url').tomlSection).toBe('defaults.multimodal')

    const advancedServerSettings = CONFIGURATION_DEFAULTS.settings.filter(
      (setting: ConfigurationDefaultsSetting) => setting.categoryId === 'advanced-server'
    ) as readonly (ConfigurationDefaultsSetting & {
      tomlSection?: string
      visibility?: 'standard' | 'advanced'
      mutability?: 'runtime' | 'restart-required'
    })[]
    expect(advancedServerSettings.map((setting) => setting.id)).toEqual(ADVANCED_SERVER_SETTING_IDS)
    expect(advancedServerSettings).toHaveLength(1)
    expect(advancedServerSettings[0].mutability).toBe('restart-required')
    expect(advancedServerSettings[0].visibility).toBe('advanced')
    expect(advancedServerSettings[0].tomlSection).toBe('defaults.advanced.server')
    expect(settingById('server-alias').control).toMatchObject({
      kind: 'text',
      name: 'alias',
      value: '',
      placeholder: 'model alias'
    })
    expect(settingById('server-alias').tomlSection).toBe('defaults.advanced.server')
    expect(settingById('server-alias').mutability).toBe('restart-required')
    expect(settingById('server-alias').visibility).toBe('advanced')
  })
})
