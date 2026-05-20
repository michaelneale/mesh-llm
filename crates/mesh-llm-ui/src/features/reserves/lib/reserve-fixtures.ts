import type { WakeableNode } from '@/features/app-tabs/types'
import {
  DEFAULT_RESERVE_WAKE_POLICY_SETTINGS,
  reservePolicyTilesFromSettings
} from '@/features/reserves/lib/reserve-policy'
import type { ReserveProvider } from '@/features/reserves/lib/reserve-types'

export const RESERVE_PROVIDER_FIXTURES: ReserveProvider[] = [
  {
    id: 'vast',
    name: 'Vast.ai',
    kind: 'Cloud GPU',
    icon: 'cloud',
    region: 'us-east · us-west · eu-central',
    tags: ['us-east', 'us-west', 'eu-central'],
    billing: '$0.34–$1.80 / GPU-hr',
    summary: 'Spot GPU capacity ready to absorb long-context and overnight burst runs.',
    nodes: [
      {
        id: 'vast-a100-1',
        hw: 'A100 80GB',
        vram: 80,
        models: ['Qwen2.5-72B-Instruct'],
        state: 'waking',
        eta: 155,
        progress: 35
      },
      {
        id: 'vast-a100-2',
        hw: 'A100 80GB',
        vram: 80,
        models: ['Qwen2.5-72B-Instruct'],
        state: 'standby'
      },
      {
        id: 'vast-h100-1',
        hw: 'H100 80GB',
        vram: 80,
        models: ['DeepSeek-R1', 'Llama-3.3-70B'],
        state: 'joining',
        eta: 18,
        progress: 82
      },
      { id: 'vast-3090-1', hw: 'RTX 3090', vram: 24, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'vast-3090-2', hw: 'RTX 3090', vram: 24, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'vast-3090-3', hw: 'RTX 3090', vram: 24, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'vast-4090-1', hw: 'RTX 4090', vram: 24, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' },
      { id: 'vast-4090-2', hw: 'RTX 4090', vram: 24, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' },
      {
        id: 'vast-a40-1',
        hw: 'A40 48GB',
        vram: 48,
        models: ['Qwen3.6-35B-A3B-UD'],
        state: 'failed',
        error: 'auth: api key rejected',
        failedAt: '22s ago',
        retryable: true
      },
      { id: 'vast-a40-2', hw: 'A40 48GB', vram: 48, models: ['Qwen3.6-35B-A3B-UD'], state: 'standby' },
      {
        id: 'vast-a40-3',
        hw: 'A40 48GB',
        vram: 48,
        models: ['Qwen3.6-35B-A3B-UD'],
        state: 'unreachable',
        error: 'provider api timeout',
        lastSeen: '4m ago',
        retryable: true
      },
      { id: 'vast-l4-1', hw: 'L4 24GB', vram: 24, models: ['Qwen3.6-27B-UD'], state: 'standby' }
    ]
  },
  {
    id: 'runpod',
    name: 'RunPod',
    kind: 'Cloud GPU',
    icon: 'cloud',
    region: 'us-east · eu-north',
    tags: ['us-east', 'eu-north'],
    billing: '$0.69–$2.49 / GPU-hr',
    summary: 'Primary elastic pool for inference spikes that outrun local VRAM.',
    nodes: [
      { id: 'runpod-h100-1', hw: 'H100 PCIe 80GB', vram: 80, models: ['DeepSeek-R1'], state: 'standby' },
      {
        id: 'runpod-h100-2',
        hw: 'H100 80GB',
        vram: 80,
        models: ['DeepSeek-R1', 'Qwen3-32B'],
        state: 'waking',
        eta: 55,
        progress: 60
      },
      { id: 'runpod-a6000-1', hw: 'A6000 48GB', vram: 48, models: ['Qwen3.6-35B-A3B-UD'], state: 'standby' },
      { id: 'runpod-a6000-2', hw: 'A6000 48GB', vram: 48, models: ['Qwen3.6-35B-A3B-UD'], state: 'standby' },
      { id: 'runpod-a6000-3', hw: 'A6000 48GB', vram: 48, models: ['Qwen3.6-35B-A3B-UD'], state: 'standby' },
      { id: 'runpod-l40-1', hw: 'L40 48GB', vram: 48, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' },
      {
        id: 'runpod-l40-2',
        hw: 'L40 48GB',
        vram: 48,
        models: ['gemma-4-26B-A4B-it-UD'],
        state: 'failed',
        error: 'quota exceeded · 0 / $50 spend cap',
        failedAt: '1m ago',
        retryable: false
      }
    ]
  },
  {
    id: 'do',
    name: 'DigitalOcean',
    kind: 'Cloud GPU',
    icon: 'cloud',
    region: 'nyc1 · sfo3 · ams3',
    tags: ['nyc1', 'sfo3', 'ams3'],
    billing: '$0.76–$2.49 / GPU-hr',
    summary: 'Reserved GPU droplets held as last-resort capacity for public demo traffic.',
    nodes: [
      { id: 'do-h100-1', hw: 'H100 80GB', vram: 80, models: ['DeepSeek-R1'], state: 'online', since: '6h' },
      { id: 'do-h100-2', hw: 'H100 80GB', vram: 80, models: ['DeepSeek-R1'], state: 'online', since: '6h' },
      { id: 'do-a100-1', hw: 'A100 80GB', vram: 80, models: ['Qwen2.5-72B-Instruct'], state: 'online', since: '2h' },
      { id: 'do-a100-2', hw: 'A100 80GB', vram: 80, models: ['Qwen2.5-72B-Instruct'], state: 'standby' },
      { id: 'do-a100-3', hw: 'A100 80GB', vram: 80, models: ['Qwen2.5-72B-Instruct'], state: 'standby' },
      { id: 'do-l40-1', hw: 'L40 48GB', vram: 48, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' },
      { id: 'do-l40-2', hw: 'L40 48GB', vram: 48, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' }
    ]
  },
  {
    id: 'lambda',
    name: 'Lambda Labs',
    kind: 'Cloud GPU',
    icon: 'cloud',
    region: 'us-east · us-west · asia-1',
    tags: ['us-east', 'us-west', 'asia-1'],
    billing: '$0.50–$2.49 / GPU-hr · reserved',
    summary: 'Fallback wake targets for large memory jobs that need fast replacement.',
    nodes: [
      {
        id: 'lambda-h200-1',
        hw: 'H200 SXM 141GB',
        vram: 141,
        models: ['DeepSeek-R1', 'Llama-3.3-70B'],
        state: 'standby'
      },
      {
        id: 'lambda-h200-2',
        hw: 'H200 SXM 141GB',
        vram: 141,
        models: ['DeepSeek-R1', 'Llama-3.3-70B'],
        state: 'standby'
      },
      { id: 'lambda-h100-1', hw: 'H100 SXM 80GB', vram: 80, models: ['DeepSeek-R1'], state: 'standby' },
      { id: 'lambda-h100-2', hw: 'H100 SXM 80GB', vram: 80, models: ['DeepSeek-R1'], state: 'standby' },
      { id: 'lambda-h100-3', hw: 'H100 SXM 80GB', vram: 80, models: ['DeepSeek-R1'], state: 'standby' },
      { id: 'lambda-a100-1', hw: 'A100 80GB', vram: 80, models: ['Qwen2.5-72B-Instruct'], state: 'standby' },
      { id: 'lambda-a100-2', hw: 'A100 80GB', vram: 80, models: ['Qwen2.5-72B-Instruct'], state: 'standby' },
      { id: 'lambda-a100-3', hw: 'A100 80GB', vram: 80, models: ['Qwen2.5-72B-Instruct'], state: 'standby' },
      { id: 'lambda-l40-1', hw: 'L40 48GB', vram: 48, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' },
      { id: 'lambda-l40-2', hw: 'L40 48GB', vram: 48, models: ['gemma-4-26B-A4B-it-UD'], state: 'standby' }
    ]
  },
  {
    id: 'metal',
    name: 'Bare metal hosts',
    kind: 'Co-located · always on',
    icon: 'server',
    region: 'rack-01 · rack-02 · syd · home-lab',
    tags: ['rack-01', 'rack-02', 'syd', 'home-lab'],
    billing: 'flat · contributing to mesh',
    summary: 'Racked GPUs that stay closest to production data paths and private models.',
    nodes: [
      {
        id: 'rack-01',
        hw: '4×A100 SXM 80GB',
        vram: 320,
        models: ['DeepSeek-R1', 'Llama-3.3-70B', 'Qwen2.5-72B-Instruct'],
        state: 'online',
        since: '14d'
      },
      {
        id: 'rack-02',
        hw: '4×A100 SXM 80GB',
        vram: 320,
        models: ['DeepSeek-R1', 'Llama-3.3-70B'],
        state: 'online',
        since: '14d'
      },
      {
        id: 'syd-h100-1',
        hw: '8×H100 SXM',
        vram: 640,
        models: ['DeepSeek-R1', 'Qwen2.5-72B-Instruct'],
        state: 'online',
        since: '3d'
      },
      { id: 'home-lab-1', hw: '2×4090', vram: 48, models: ['Qwen3.6-27B-UD'], state: 'online', since: '31d' },
      { id: 'home-lab-2', hw: '2×3090', vram: 48, models: ['Qwen3.6-27B-UD'], state: 'online', since: '31d' }
    ]
  },
  {
    id: 'lan',
    name: 'Office LAN',
    kind: 'On-prem · mesh-llm not started',
    icon: 'lan',
    region: '192.168.1.0/24 · discovered via mDNS',
    tags: ['192.168.1.0/24', 'mDNS', 'idle workstations'],
    billing: 'no cost · workstation idle',
    summary: 'Operator-controlled desktops and workstations that wake first for the cheapest burst capacity.',
    nodes: [
      {
        id: 'design-mbp-01',
        hw: 'M3 Max · 64 GB',
        vram: 48,
        models: ['Qwen3.6-27B-UD', 'gemma-4-26B-A4B-it-UD'],
        state: 'standby'
      },
      { id: 'design-mbp-02', hw: 'M3 Max · 36 GB', vram: 27, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'design-mbp-03', hw: 'M2 Max · 32 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-eng-04', hw: 'M4 Pro · 48 GB', vram: 36, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'ws-eng-05', hw: 'M4 Pro · 48 GB', vram: 36, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'ws-eng-06', hw: 'M4 Pro · 48 GB', vram: 36, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'ws-eng-07', hw: 'RTX 4090 · 24 GB', vram: 24, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'ws-eng-08', hw: 'RTX 4090 · 24 GB', vram: 24, models: ['Qwen3.6-27B-UD'], state: 'standby' },
      { id: 'ws-eng-09', hw: 'RTX 3090 · 24 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-eng-10', hw: 'RTX 3090 · 24 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-eng-11', hw: 'RTX 3090 · 24 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-eng-12', hw: 'RTX 3090 · 24 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-data-01', hw: 'RTX 4080 · 16 GB', vram: 16, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-data-02', hw: 'RTX 4080 · 16 GB', vram: 16, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-prod-01', hw: 'RTX 4080 · 16 GB', vram: 16, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-prod-02', hw: 'RTX 4070 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-prod-03', hw: 'RTX 4070 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-mkt-01', hw: 'M2 Pro · 32 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-mkt-02', hw: 'M2 Pro · 32 GB', vram: 24, models: ['Qwen3.5-4B-UD'], state: 'standby' },
      { id: 'ws-mkt-03', hw: 'M1 Pro · 16 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      {
        id: 'ws-mkt-04',
        hw: 'M1 Pro · 16 GB',
        vram: 12,
        models: ['Qwen3.5-2B'],
        state: 'unreachable',
        error: 'offline',
        lastSeen: '2h ago',
        retryable: true
      },
      { id: 'ws-mkt-05', hw: 'M1 Pro · 16 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-ops-01', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-ops-02', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-ops-03', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-ops-04', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-ops-05', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-qa-01', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-qa-02', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' },
      { id: 'ws-qa-03', hw: 'RTX 3060 · 12 GB', vram: 12, models: ['Qwen3.5-2B'], state: 'standby' }
    ]
  }
]

export const RESERVE_POLICY_TILES = reservePolicyTilesFromSettings(DEFAULT_RESERVE_WAKE_POLICY_SETTINGS)

export function reserveProvidersFromWakeableNodes(
  wakeableNodes: WakeableNode[] | undefined
): ReserveProvider[] | undefined {
  if (!wakeableNodes) return undefined

  const providers = new Map<string, ReserveProvider>()

  for (const node of wakeableNodes) {
    const providerName = node.provider ?? 'Unassigned reserves'
    const providerId = providerName.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    const provider = providers.get(providerId) ?? {
      id: providerId,
      name: providerName,
      kind: node.provider ? 'Provider' : 'Reserve pool',
      region: 'pending',
      tags: ['pending'],
      billing: 'not configured',
      summary: 'Live wakeable node advertisement awaiting richer provider metadata.',
      nodes: []
    }

    provider.nodes.push({
      id: node.logical_id,
      hw: 'Advertised reserve node',
      vram: node.vram_gb,
      location: node.provider ? `${providerName} reserve` : 'advertised reserve',
      note: 'Live node metadata is limited to the wakeable-node status payload.',
      models: node.models,
      state: node.state === 'waking' ? 'waking' : 'standby',
      eta: node.wake_eta_secs
    })
    providers.set(providerId, provider)
  }

  return [...providers.values()]
}
