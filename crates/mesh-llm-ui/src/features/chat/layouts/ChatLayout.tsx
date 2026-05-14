import * as Popover from '@radix-ui/react-popover'
import { MessageSquare } from 'lucide-react'
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
  type ReactNode,
  type RefObject
} from 'react'

const DESKTOP_SIDEBAR_QUERY = '(min-width: 1024px)'
const CHAT_LAYOUT_FALLBACK_HEIGHT = 'calc(100dvh - 180px)'
const CHAT_LAYOUT_MIN_HEIGHT = 320
const CHAT_LAYOUT_KEYBOARD_GUTTER = 12

type ChatLayoutStyle = CSSProperties & {
  '--chat-layout-height': string
}

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

function readChatLayoutHeight(layoutElement: HTMLDivElement, viewport: VisualViewport) {
  const layoutRect = layoutElement.getBoundingClientRect()
  const topInsideVisualViewport = Math.max(0, layoutRect.top - viewport.offsetTop)
  const availableHeight = viewport.height - topInsideVisualViewport - CHAT_LAYOUT_KEYBOARD_GUTTER

  return `${Math.max(CHAT_LAYOUT_MIN_HEIGHT, Math.floor(availableHeight))}px`
}

function useChatVisualViewportHeight(layoutRef: RefObject<HTMLDivElement | null>) {
  const [layoutHeight, setLayoutHeight] = useState(CHAT_LAYOUT_FALLBACK_HEIGHT)

  useLayoutEffect(() => {
    const viewport = window.visualViewport
    if (!viewport) return undefined

    let animationFrame: number | undefined

    const updateLayoutHeight = () => {
      const layoutElement = layoutRef.current
      if (!layoutElement) return
      setLayoutHeight(readChatLayoutHeight(layoutElement, viewport))
    }

    const scheduleUpdate = () => {
      if (animationFrame !== undefined) window.cancelAnimationFrame(animationFrame)
      animationFrame = window.requestAnimationFrame(updateLayoutHeight)
    }

    updateLayoutHeight()
    viewport.addEventListener('resize', scheduleUpdate)
    viewport.addEventListener('scroll', scheduleUpdate)
    window.addEventListener('resize', scheduleUpdate)
    window.addEventListener('orientationchange', scheduleUpdate)

    return () => {
      if (animationFrame !== undefined) window.cancelAnimationFrame(animationFrame)
      viewport.removeEventListener('resize', scheduleUpdate)
      viewport.removeEventListener('scroll', scheduleUpdate)
      window.removeEventListener('resize', scheduleUpdate)
      window.removeEventListener('orientationchange', scheduleUpdate)
    }
  }, [layoutRef])

  return layoutHeight
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
  const layoutRef = useRef<HTMLDivElement | null>(null)
  const messageListRef = useRef<HTMLDivElement | null>(null)
  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const chatLayoutHeight = useChatVisualViewportHeight(layoutRef)
  const desktopSidebarViewport = useDesktopSidebarViewport()
  const showDesktopSidebar =
    !hideSidebar && (sidebarMode === 'desktop' || (sidebarMode === 'auto' && desktopSidebarViewport))
  const handleMessageAreaPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onMessageAreaClick?.()
    }
  }

  const chatLayoutStyle: ChatLayoutStyle = {
    '--chat-layout-height': chatLayoutHeight,
    height: 'var(--chat-layout-height)',
    maxHeight: 'var(--chat-layout-height)'
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
      ref={layoutRef}
      className={
        hideSidebar
          ? 'relative grid min-h-0 min-w-0 items-stretch gap-4 overflow-hidden'
          : 'relative grid min-h-0 min-w-0 items-stretch gap-4 overflow-hidden lg:grid-cols-[minmax(240px,28vw)_minmax(0,1fr)] xl:grid-cols-[minmax(280px,320px)_minmax(0,1fr)]'
      }
      data-testid="chat-layout"
      style={chatLayoutStyle}
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
          <div className="border-t border-border-soft bg-panel px-4 pb-[calc(0.75rem+env(safe-area-inset-bottom))] pt-3 sm:py-3">
            {composer}
          </div>
        </div>
      </section>
      {!hideSidebar && !showDesktopSidebar ? (
        <Popover.Root>
          <Popover.Trigger asChild>
            <button
              aria-label="Open chat sidebar"
              className="ui-control-primary fixed bottom-[calc(1.25rem+env(safe-area-inset-bottom))] right-[calc(1.25rem+env(safe-area-inset-right))] z-30 inline-flex size-[55px] items-center justify-center rounded-full border shadow-surface-popover outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
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
