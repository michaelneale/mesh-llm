import { describe, expect, it } from 'vitest'
import type { ModelSummary } from '@/features/app-tabs/types'
import {
  buildModelFilterOptions,
  modelFamilyLabel,
  modelFilterValue,
  modelLicenseLabel,
  modelProviderLabel,
  modelVariantLabel
} from '@/features/network/lib/model-catalog-utils'

function model(overrides: Partial<ModelSummary>): ModelSummary {
  return {
    name: 'Unknown-Model',
    family: 'unknown',
    size: '7B',
    context: '32K',
    status: 'ready',
    tags: [],
    ...overrides
  }
}

describe('model catalog filter utilities', () => {
  it.each([
    ['Qwen3-235B-A22B-Instruct-2507', 'Qwen'],
    ['Qwen/Qwen3-VL-8B-Instruct', 'Qwen'],
    ['qwen-max', 'Qwen'],
    ['QwQ-32B', 'Qwen'],
    ['openai/gpt-oss-120b', 'GPT-OSS'],
    ['gpt-oss-20b', 'GPT-OSS'],
    ['command-a-03-2025', 'Command'],
    ['GLM-4.7', 'GLM'],
    ['glm-4.6v', 'GLM'],
    ['zai-org/GLM-4.5-Air', 'GLM'],
    ['step-3', 'Step'],
    ['llama-3.1-nemotron-ultra-253b-v1', 'Nemotron'],
    ['nvidia-llama-3.3-nemotron-super-49b-v1.5', 'Nemotron'],
    ['ling-flash-2.0', 'Ling'],
    ['meta-llama/Llama-3.3-70B-Instruct', 'Llama'],
    ['mistralai/Mistral-Small-3.2-24B-Instruct-2506', 'Mistral'],
    ['mistralai/Mixtral-8x7B-Instruct-v0.1', 'Mixtral'],
    ['google/gemma-3-27b-it', 'Gemma'],
    ['google/gemma-3n-e4b-it', 'Gemma'],
    ['deepseek-ai/DeepSeek-R1-0528', 'DeepSeek'],
    ['DS-R1-Distill-Qwen-32B', 'DeepSeek'],
    ['microsoft/Phi-4-mini-instruct', 'Phi'],
    ['01-ai/Yi-34B-Chat', 'Yi'],
    ['ibm-granite/granite-3.3-8b-instruct', 'Granite'],
    ['bigcode/starcoder2-15b', 'StarCoder'],
    ['internlm/internlm3-8b-instruct', 'InternLM'],
    ['THUDM/CogVLM2-llama3-chat-19B', 'Llama']
  ])('normalizes encoded family from %s to %s', (name, expected) => {
    expect(modelFamilyLabel(model({ name, fullId: name }))).toBe(expected)
  })

  it('falls back to a normalized family token when no known rule matches', () => {
    expect(modelFamilyLabel(model({ name: 'acme-2.1-chat', fullId: 'org/acme-2.1-chat' }))).toBe('Acme')
  })

  it('uses normalized family labels for family filter options', () => {
    const models = [
      model({ name: 'Qwen3-8B-Instruct', fullId: 'Qwen/Qwen3-8B-Instruct' }),
      model({ name: 'Qwen2.5-Coder-32B-Instruct', fullId: 'Qwen/Qwen2.5-Coder-32B-Instruct' }),
      model({ name: 'GLM-4.7', fullId: 'zai-org/GLM-4.7' })
    ]

    expect(buildModelFilterOptions(models, 'family')).toEqual([
      { value: 'GLM', count: 1 },
      { value: 'Qwen', count: 2 }
    ])
    expect(modelFilterValue(models[0], 'family')).toBe('Qwen')
  })

  it('uses normalized family labels for provider detection', () => {
    expect(modelProviderLabel(model({ name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M' }))).toBe('Mistral AI')
    expect(modelProviderLabel(model({ name: 'gpt-oss-120b', fullId: 'openai/gpt-oss-120b' }))).toBe('OpenAI')
    expect(modelProviderLabel(model({ name: 'command-a-03-2025' }))).toBe('Cohere')
    expect(modelProviderLabel(model({ name: 'glm-4.5v' }))).toBe('Z.ai')
    expect(modelProviderLabel(model({ name: 'step-3' }))).toBe('StepFun')
    expect(modelProviderLabel(model({ name: 'llama-3.1-nemotron-ultra-253b-v1' }))).toBe('Nvidia')
    expect(modelProviderLabel(model({ name: 'ling-flash-2.0' }))).toBe('Ant Group')
    expect(
      modelProviderLabel(model({ name: 'MiniMax-M2.5-GGUF:Q4_K_M', fullId: 'unsloth/MiniMax-M2.5-GGUF:Q4_K_M' }))
    ).toBe('MiniMax')
  })

  it.each([
    ['command-a-03-2025', 'CC-BY-NC-4.0'],
    ['glm-4.5v', 'MIT'],
    ['gpt-oss-120b', 'Apache 2.0'],
    ['step-3', 'Apache 2.0'],
    ['qwen3-32b', 'Apache 2.0'],
    ['llama-3.1-nemotron-ultra-253b-v1', 'Nvidia Open Model'],
    ['ling-flash-2.0', 'MIT'],
    ['minimax-m2', 'Apache 2.0'],
    ['nvidia-llama-3.3-nemotron-super-49b-v1.5', 'Nvidia Open'],
    ['gemma-3-12b-it', 'Gemma'],
    ['qwq-32b', 'Apache 2.0'],
    ['llama-3.1-405b-instruct-bf16', 'Llama 3.1 Community'],
    ['llama-3.1-405b-instruct-fp8', 'Llama 3.1 Community']
  ])('infers chart license for %s as %s', (name, expected) => {
    expect(modelLicenseLabel(model({ name, fullId: name }))).toBe(expected)
    expect(modelFilterValue(model({ name, fullId: name }), 'license')).toBe(expected)
  })

  it('uses explicit model license before inferred chart rules', () => {
    expect(modelLicenseLabel(model({ name: 'qwen3-32b', license: 'Tongyi Qianwen' }))).toBe('Tongyi Qianwen')
  })

  it('builds license filter options from inferred and explicit licenses', () => {
    const models = [
      model({ name: 'qwen3-32b' }),
      model({ name: 'gpt-oss-120b' }),
      model({ name: 'glm-4.5v' }),
      model({ name: 'custom-1', license: 'Research Only' })
    ]

    expect(buildModelFilterOptions(models, 'license')).toEqual([
      { value: 'Apache 2.0', count: 2 },
      { value: 'MIT', count: 1 },
      { value: 'Research Only', count: 1 }
    ])
  })

  it('uses the publisher segment as the variant filter value', () => {
    const models = [
      model({ name: 'Qwen3-8B-Instruct', fullId: 'unsloth/Qwen3-8B-Instruct-GGUF' }),
      model({ name: 'Qwen3-8B-Instruct', fullId: 'bartowski/Qwen3-8B-Instruct-GGUF' }),
      model({ name: 'Qwen3-4B-Q4_K_M', fullId: 'Qwen3-4B-Q4_K_M' }),
      model({ name: 'GLM-4.7' })
    ]

    expect(modelVariantLabel(models[0])).toBe('Unsloth')
    expect(modelVariantLabel(models[2])).toBe('—')
    expect(buildModelFilterOptions(models, 'variant')).toEqual([
      { value: '—', count: 2 },
      { value: 'Bartowski', count: 1 },
      { value: 'Unsloth', count: 1 }
    ])
    expect(modelFilterValue(models[1], 'variant')).toBe('Bartowski')
  })
})
