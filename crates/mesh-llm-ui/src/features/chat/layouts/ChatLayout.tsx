import * as Popover from '@radix-ui/react-popover'
import { MessageSquare } from 'lucide-react'
import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react'

const DESKTOP_SIDEBAR_QUERY = '(min-width: 1024px)'

type ChatLayoutProps = {
  sidebar: ReactNode
  sidebarMode?: 'auto' | 'desktop' | 'compact'
  hideSidebar?: boolean
  title: string
  subtitle?: ReactNode
  actions: ReactNode
  children: ReactNode
  composer: ReactNode
  onMessageAreaClick?: () => void
}

function shouldUseDesktopSidebarFallback() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return true
  if (import.meta.env.VITEST || import.meta.env.MODE === 'test') return true
  if (typeof navigator !== 'undefined' && navigator.userAgent.includes('jsdom')) return true
  return false
}

function readDesktopSidebarViewport() {
  if (shouldUseDesktopSidebarFallback()) return true
  return window.matchMedia(DESKTOP_SIDEBAR_QUERY).matches
}

function useDesktopSidebarViewport() {
  const [matches, setMatches] = useState(readDesktopSidebarViewport)

  useEffect(() => {
    if (shouldUseDesktopSidebarFallback()) return undefined

    const mediaQuery = window.matchMedia(DESKTOP_SIDEBAR_QUERY)
    const handleChange = (event: MediaQueryListEvent) => setMatches(event.matches)

    mediaQuery.addEventListener('change', handleChange)

    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return matches
}

export function ChatLayout({
  sidebar,
  sidebarMode = 'auto',
  hideSidebar = false,
  title,
  subtitle,
  actions,
  children,
  composer,
  onMessageAreaClick
}: ChatLayoutProps) {
  const messageListRef = useRef<HTMLDivElement | null>(null)
  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const desktopSidebarViewport = useDesktopSidebarViewport()
  const showDesktopSidebar =
    !hideSidebar && (sidebarMode === 'desktop' || (sidebarMode === 'auto' && desktopSidebarViewport))
  const handleMessageAreaPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onMessageAreaClick?.()
    }
  }

  useLayoutEffect(() => {
    const messageList = messageListRef.current
    if (!messageList) return undefined

    const scrollToBottom = () => {
      messageList.scrollTop = messageList.scrollHeight
      messageEndRef.current?.scrollIntoView({ block: 'end' })
    }

    scrollToBottom()

    if (typeof window.requestAnimationFrame !== 'function') return undefined

    const frame = window.requestAnimationFrame(scrollToBottom)
    return () => window.cancelAnimationFrame(frame)
  })

  return (
    <div
      className={
        hideSidebar
          ? 'relative grid min-h-0 min-w-0 items-stretch gap-4 overflow-hidden'
          : 'relative grid min-h-0 min-w-0 items-stretch gap-4 overflow-hidden lg:grid-cols-[minmax(240px,28vw)_minmax(0,1fr)] xl:grid-cols-[minmax(280px,320px)_minmax(0,1fr)]'
      }
      data-testid="chat-layout"
      style={{ height: 'calc(100dvh - 180px)', maxHeight: 'calc(100dvh - 180px)' }}
    >
      {showDesktopSidebar ? <div className="min-h-0 min-w-0 overflow-hidden [&>*]:h-full">{sidebar}</div> : null}
      <section className="panel-shell flex min-h-0 min-w-0 select-none flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
        <header className="flex flex-wrap items-start justify-between gap-2 border-b border-border-soft px-3.5 py-2.5 md:flex-nowrap">
          <div className="flex min-w-0 shrink-0 items-center self-stretch md:block md:self-auto">
            <h1 className="text-[length:var(--density-type-control)] font-semibold tracking-[0.02em]">{title}</h1>
            {subtitle ? (
              <div className="mt-0.5 hidden text-[length:var(--density-type-label)] text-fg-faint md:block">
                {subtitle}
              </div>
            ) : null}
          </div>
          <div className="ml-3 flex min-w-0 flex-1 flex-wrap items-center justify-start gap-2 md:ml-0 md:justify-end">
            {actions}
          </div>
        </header>
        <div className="flex min-h-0 flex-1 flex-col">
          <div
            ref={messageListRef}
            className="chat-message-scrollbar min-h-0 flex-1 overflow-x-hidden overflow-y-auto px-4 py-4 sm:px-[26px] sm:py-5"
            data-testid="chat-message-list"
            onPointerDown={handleMessageAreaPointerDown}
          >
            {children}
            <div aria-hidden={true} data-chat-scroll-anchor="true" ref={messageEndRef} />
          </div>
          <div className="border-t border-border-soft bg-panel px-4 py-3">{composer}</div>
        </div>
      </section>
      {!hideSidebar && !showDesktopSidebar ? (
        <Popover.Root>
          <Popover.Trigger asChild>
            <button
              aria-label="Open chat sidebar"
              className="ui-control-primary fixed bottom-5 right-5 z-30 inline-flex size-[55px] items-center justify-center rounded-full border shadow-surface-popover outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              type="button"
            >
              <MessageSquare aria-hidden={true} className="size-[20px]" strokeWidth={1.7} />
            </button>
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content
              align="end"
              className="z-50 h-[min(72vh,42rem)] w-[min(24rem,calc(100vw-2rem))] overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel shadow-surface-popover outline-none [&>*]:h-full"
              collisionPadding={12}
              side="top"
              sideOffset={10}
            >
              {sidebar}
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      ) : null}
    </div>
  )
}
