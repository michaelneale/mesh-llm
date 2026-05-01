import type { PointerEvent, ReactNode } from 'react'

type ChatLayoutProps = {
  sidebar: ReactNode
  title: string
  subtitle?: ReactNode
  actions: ReactNode
  children: ReactNode
  composer: ReactNode
  onMessageAreaClick?: () => void
}

export function ChatLayout({ sidebar, title, subtitle, actions, children, composer, onMessageAreaClick }: ChatLayoutProps) {
  const handleMessageAreaPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onMessageAreaClick?.()
    }
  }

  return (
    <div className="grid items-stretch gap-4" style={{ gridTemplateColumns: '360px 1fr', minHeight: 'calc(100vh - 180px)' }}>
      {sidebar}
    <section className="panel-shell flex min-h-0 select-none flex-col overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel">
        <header className="flex items-center justify-between border-b border-border-soft px-3.5 py-2.5">
          <div>
            <h1 className="text-[length:var(--density-type-control)] font-semibold tracking-[0.02em]">{title}</h1>
            {subtitle ? <div className="mt-0.5 text-[length:var(--density-type-label)] text-fg-faint">{subtitle}</div> : null}
          </div>
          <div className="flex items-center gap-2">
            {actions}
          </div>
        </header>
        <div className="flex flex-1 flex-col">
          <div className="flex-1 overflow-auto px-[26px] py-5" data-testid="chat-message-list" onPointerDown={handleMessageAreaPointerDown}>
            {children}
          </div>
          <div className="border-t border-border-soft bg-panel px-4 py-3">
            {composer}
          </div>
        </div>
      </section>
    </div>
  )
}
