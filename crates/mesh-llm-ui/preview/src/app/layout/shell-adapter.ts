import { SHELL_HARNESS } from '@/features/app-tabs/data'
import type { LinkItem, ShellHarnessData, TopNavJoinCommand } from '@/features/app-tabs/types'
import type { StatusPayload } from '@/lib/api/types'
import { env } from '@/lib/env'
import { isPublicMesh } from '@/lib/api/mesh-visibility'

export type TopNavShellData = {
  apiUrl: string
  topNavApiAccessLinks: LinkItem[]
  topNavJoinCommands: TopNavJoinCommand[]
  topNavJoinLinks: LinkItem[]
}

function normalizeOpenAIBaseUrl(value: string) {
  const trimmed = value.trim().replace(/\/+$/, '')
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`
}

function resolveOpenAIBaseUrl(status?: StatusPayload) {
  if (
    typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ) {
    return normalizeOpenAIBaseUrl(`http://127.0.0.1:${status?.api_port ?? 9337}`)
  }

  if (env.isDevelopment) {
    return normalizeOpenAIBaseUrl(env.apiUrl)
  }

  return normalizeOpenAIBaseUrl(env.apiUrl)
}

function resolveInviteToken(status?: StatusPayload) {
  return status?.token?.trim() || null
}

function buildAvailableJoinCommands(inviteToken: string): TopNavJoinCommand[] {
  return [
    {
      label: 'Invite token',
      value: inviteToken,
      hint: 'Use the issued live token from /api/status.',
      noWrapValue: true
    },
    {
      label: 'Auto join and serve command',
      value: `mesh-llm --auto --join ${inviteToken}`,
      prefix: '$',
      hint: 'Matches the Connect panel flow: join, select a model, and serve the API.'
    },
    { label: 'Client-only join command', value: `mesh-llm client --join ${inviteToken}`, prefix: '$' },
    { label: 'Blackboard skill command', value: 'mesh-llm blackboard install-skill', prefix: '$' }
  ]
}

function buildPublicJoinCommands(): TopNavJoinCommand[] {
  return [
    {
      label: 'Public mesh command',
      value: 'mesh-llm --auto',
      prefix: '$',
      hint: 'Join public discovery, auto-select a model, and serve the local API.'
    },
    { label: 'Blackboard skill command', value: 'mesh-llm blackboard install-skill', prefix: '$' }
  ]
}

function buildUnavailableJoinCommands(): TopNavJoinCommand[] {
  return [
    {
      label: 'Invite token',
      value: 'Invite token unavailable',
      hint: 'Live /api/status has not reported an invite token yet.',
      noWrapValue: true,
      disabled: true
    },
    {
      label: 'Auto join and serve command',
      value: 'Auto join command unavailable',
      prefix: '$',
      hint: 'This command becomes available after the backend issues a live invite token.',
      disabled: true
    },
    {
      label: 'Client-only join command',
      value: 'Client-only join command unavailable',
      prefix: '$',
      hint: 'This command becomes available after the backend issues a live invite token.',
      disabled: true
    },
    { label: 'Blackboard skill command', value: 'mesh-llm blackboard install-skill', prefix: '$' }
  ]
}

export function resolveHarnessTopNavData(data: ShellHarnessData): TopNavShellData {
  return {
    apiUrl: env.apiUrl,
    topNavApiAccessLinks: data.topNavApiAccessLinks,
    topNavJoinCommands: data.topNavJoinCommands,
    topNavJoinLinks: data.topNavJoinLinks
  }
}

export function resolveLiveTopNavData(status?: StatusPayload): TopNavShellData {
  const inviteToken = resolveInviteToken(status)
  const topNavJoinCommands =
    status && isPublicMesh(status)
      ? buildPublicJoinCommands()
      : inviteToken
        ? buildAvailableJoinCommands(inviteToken)
        : buildUnavailableJoinCommands()

  return {
    apiUrl: resolveOpenAIBaseUrl(status),
    topNavApiAccessLinks: SHELL_HARNESS.topNavApiAccessLinks,
    topNavJoinCommands,
    topNavJoinLinks: SHELL_HARNESS.topNavJoinLinks
  }
}
