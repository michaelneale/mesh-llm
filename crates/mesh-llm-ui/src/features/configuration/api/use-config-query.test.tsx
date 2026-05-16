// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'

import { useConfigQuery } from '@/features/configuration/api/use-config-query'
import type { ModelsResponse, StatusPayload } from '@/lib/api/types'

const STATUS_PAYLOAD: StatusPayload = {
  node_id: 'self',
  node_state: 'serving',
  model_name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
  peers: [],
  models: [],
  my_vram_gb: 0,
  gpus: [],
  serving_models: []
}

const MODELS_RESPONSE: ModelsResponse = {
  mesh_models: [
    {
      name: 'Hermes-2-Pro-Mistral-7B-Q4_K_M',
      status: 'warm',
      node_count: 1,
      quantization: 'Q4_K_M'
    }
  ]
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('useConfigQuery', () => {
  it('keeps local defaults visible while runtime-control hydration is still loading', async () => {
    const getConfigDeferred = createDeferredResponse({
      snapshot: {
        revision: 7,
        config: {
          version: 1,
          defaults: {
            request_defaults: {
              reasoning_format: 'qwen'
            }
          }
        }
      }
    })

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: true,
          local_only: true,
          requires_explicit_remote_endpoint: false,
          endpoint: 'control://owner'
        })
      }
      if (url.endsWith('/api/runtime/control/get-config')) {
        expect(JSON.parse(String(init?.body))).toEqual({ endpoint: 'control://owner' })
        return getConfigDeferred.promise
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() => expect(result.current.data).toBeDefined())

    expect(readSettingValue(result.current.data!, 'reasoning-format')).toBe('deepseek')

    getConfigDeferred.resolve(jsonResponse(getConfigDeferred.body))

    await waitFor(() => expect(readSettingValue(result.current.data!, 'reasoning-format')).toBe('qwen'))
    expect(fetchMock).toHaveBeenCalledWith('/api/runtime/control-bootstrap')
  })

  it('keeps the local defaults fallback when bootstrap is disabled', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: false,
          local_only: true,
          requires_explicit_remote_endpoint: true,
          disabled_reason: 'missing_owner_identity',
          message: 'Configuration saving requires a local owner identity.',
          suggested_commands: [
            'mesh-llm auth status',
            'mesh-llm auth init --no-passphrase',
            'mesh-llm serve --owner-required'
          ]
        })
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() => expect(result.current.data).toBeDefined())

    expect(readSettingValue(result.current.data!, 'reasoning-format')).toBe('deepseek')
    expect(fetchMock).not.toHaveBeenCalledWith(
      '/api/runtime/control/get-config',
      expect.objectContaining({ method: 'POST' })
    )
    expect(result.current.isError).toBe(false)
    expect(result.current.controlConfigQuery.data?.bootstrap).toMatchObject({
      enabled: false,
      disabled_reason: 'missing_owner_identity',
      suggested_commands: expect.arrayContaining(['mesh-llm auth init --no-passphrase'])
    })
    expect(result.current.controlConfigQuery.data?.snapshot).toBeUndefined()
  })

  it('applies full mesh config updates with expected revision and preserved fields', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)

      if (url.endsWith('/api/status')) return jsonResponse(STATUS_PAYLOAD)
      if (url.endsWith('/api/models')) return jsonResponse(MODELS_RESPONSE)
      if (url.endsWith('/api/runtime/control-bootstrap')) {
        return jsonResponse({
          enabled: true,
          local_only: true,
          requires_explicit_remote_endpoint: false,
          endpoint: 'control://owner'
        })
      }
      if (url.endsWith('/api/runtime/control/get-config')) {
        return jsonResponse({
          snapshot: {
            revision: 7,
            config: {
              version: 1,
              owner_control: {
                bind: '127.0.0.1:7447'
              },
              telemetry: {
                enabled: true
              },
              models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
              plugin: [{ name: 'telemetry', enabled: true }],
              defaults: {
                threads: 6,
                request_defaults: {
                  temperature: 0.8,
                  reasoning_format: 'deepseek',
                  top_k: 40
                },
                speculative: {
                  draft_max_tokens: 16
                },
                skippy: {
                  activation_wire_dtype: 'f16'
                },
                multimodal: {
                  image_min_tokens: 32
                },
                advanced: {
                  server: {
                    alias: 'existing-alias'
                  }
                }
              }
            }
          }
        })
      }
      if (url.endsWith('/api/runtime/control/apply-config')) {
        const body = JSON.parse(String(init?.body)) as {
          endpoint: string
          expected_revision: number
          config: Record<string, unknown>
        }

        expect(body.endpoint).toBe('control://owner')
        expect(body.expected_revision).toBe(7)
        expect(body.config).toMatchObject({
          version: 1,
          owner_control: {
            bind: '127.0.0.1:7447'
          },
          telemetry: {
            enabled: true
          },
          models: [{ model: 'hf://meshllm/base@main:Q4_K_M', ctx_size: 8192 }],
          plugin: [{ name: 'telemetry', enabled: true }],
          defaults: {
            threads: 6,
            request_defaults: {
              temperature: 0.8,
              reasoning_format: 'qwen',
              top_k: 55
            },
            speculative: {
              draft_max_tokens: 20
            },
            skippy: {
              activation_wire_dtype: 'q8'
            },
            multimodal: {
              image_min_tokens: 64
            },
            advanced: {
              server: {
                alias: 'carrack-mesh'
              }
            }
          }
        })

        return jsonResponse({
          success: true,
          current_revision: 8,
          config_hash: 'abc123',
          apply_mode: 'live'
        })
      }

      throw new Error(`Unexpected fetch request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)

    const { result } = renderHook(() => useConfigQuery({ enabled: true }), {
      wrapper: createWrapper()
    })

    await waitFor(() => expect(readSettingValue(result.current.data!, 'reasoning-format')).toBe('deepseek'))

    const nextDefaults = readDefaultsValues(result.current.data!)
    nextDefaults['reasoning-format'] = 'qwen'
    nextDefaults['top-k'] = '55'
    nextDefaults['draft-max-tokens'] = '20'
    nextDefaults['activation-wire-dtype'] = 'q8'
    nextDefaults['image-min-tokens'] = '64'
    nextDefaults['server-alias'] = 'carrack-mesh'

    await act(async () => {
      const response = await result.current.applyDefaults(nextDefaults)
      expect(response).toMatchObject({
        success: true,
        current_revision: 8,
        apply_mode: 'live'
      })
    })

    await waitFor(() => expect(readSettingValue(result.current.data!, 'reasoning-format')).toBe('qwen'))
    expect(readSettingValue(result.current.data!, 'top-k')).toBe('55')
    expect(readSettingValue(result.current.data!, 'activation-wire-dtype')).toBe('q8')
    expect(readSettingValue(result.current.data!, 'image-min-tokens')).toBe('64')
    expect(readSettingValue(result.current.data!, 'server-alias')).toBe('carrack-mesh')
    expect(result.current.controlConfigQuery.data?.snapshot?.revision).toBe(8)
  })
})

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

function readSettingValue(data: NonNullable<ReturnType<typeof useConfigQuery>['data']>, settingId: string) {
  return data.defaults.settings.find((setting) => setting.id === settingId)?.control.value
}

function readDefaultsValues(data: NonNullable<ReturnType<typeof useConfigQuery>['data']>) {
  return Object.fromEntries(data.defaults.settings.map((setting) => [setting.id, setting.control.value]))
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  })
}

function createDeferredResponse(body: unknown) {
  let resolve: (response: Response) => void = () => undefined
  const promise = new Promise<Response>((promiseResolve) => {
    resolve = promiseResolve
  })

  return {
    body,
    promise,
    resolve
  }
}
