import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { loadChatState } from '@/features/chat/api/chat-storage'
import { ChatPage } from '@/features/chat/pages/ChatPage'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'
import { FeatureFlagProvider } from '@/lib/feature-flags'

vi.mock('@/features/chat/api/chat-storage', () => ({
  clearChatState: vi.fn(),
  loadChatState: vi.fn(),
  saveChatState: vi.fn(),
}))

function renderChatPage({ transparencyTabEnabled = false }: { transparencyTabEnabled?: boolean } = {}) {
  if (transparencyTabEnabled) {
    window.localStorage.setItem(APP_STORAGE_KEYS.featureFlagOverrides, JSON.stringify({ chat: { transparencyTab: true } }))
  }

  render(
    <FeatureFlagProvider>
      <ChatPage />
    </FeatureFlagProvider>,
  )
}

function queryAllByTextContent(text: string) {
  return screen.queryAllByText((_, element) => element?.textContent?.includes(text) ?? false, { selector: 'span' })
}

describe('ChatPage', () => {
  beforeEach(() => {
    window.localStorage.removeItem(APP_STORAGE_KEYS.featureFlagOverrides)
    vi.mocked(loadChatState).mockResolvedValue(undefined)
  })

  it('deselects the inspected message when the message area background is clicked', async () => {
    const user = userEvent.setup()

    renderChatPage({ transparencyTabEnabled: true })

    await user.click(screen.getByRole('button', { name: 'Inspect transparency' }))

    expect(screen.queryByText('No message selected')).not.toBeInTheDocument()

    await user.click(screen.getByTestId('chat-message-list'))

    expect(screen.getByText('No message selected')).toBeInTheDocument()
  })

  it('falls back to harness conversations when persisted chat state is malformed', async () => {
    vi.mocked(loadChatState).mockResolvedValue({} as Awaited<ReturnType<typeof loadChatState>>)

    renderChatPage()

    expect(await screen.findAllByText('Routing latency notes')).not.toHaveLength(0)
  })

  it('hides the transparency tab by default', () => {
    renderChatPage()

    expect(screen.queryByRole('tab', { name: /transparency/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Inspect transparency' })).not.toBeInTheDocument()
  })

  it('hides route disclosure text by default', () => {
    renderChatPage()

    expect(queryAllByTextContent('sent to lemony-28')).toHaveLength(0)
    expect(queryAllByTextContent('sent to carrack')).toHaveLength(0)
    expect(queryAllByTextContent('routed via carrack')).toHaveLength(0)
  })

  it('shows the transparency tab when the feature flag is enabled', () => {
    renderChatPage({ transparencyTabEnabled: true })

    expect(screen.getByRole('tab', { name: /transparency/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Inspect transparency' })).toBeInTheDocument()
    expect(queryAllByTextContent('sent to lemony-28').length).toBeGreaterThan(0)
    expect(queryAllByTextContent('sent to carrack').length).toBeGreaterThan(0)
    expect(queryAllByTextContent('routed via carrack').length).toBeGreaterThan(0)
  })
})
