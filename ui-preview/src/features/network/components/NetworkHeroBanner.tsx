import type { HeroAction } from '@/features/app-tabs/types'
import { GitHubIcon } from '@/components/icons/GitHubIcon'
import { InfoBanner } from '@/components/ui/InfoBanner'

function MeshIcon() {
  return (
    <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="5" cy="6" r="2.2" />
      <circle cx="19" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M6.8 7.3L10.7 16.3M17.2 7.3L13.3 16.3M7 6h10" />
    </svg>
  )
}

type NetworkHeroBannerProps = { title: string; description: string; actions: HeroAction[]; leadingIcon?: React.ReactNode }

export function NetworkHeroBanner({ title, description, actions, leadingIcon }: NetworkHeroBannerProps) {
  return (
    <InfoBanner
      action={(
        <div className="flex items-center gap-3">
        {actions.map((action) => {
          if (action.tone === 'link') {
            return <a key={action.label} className="ui-link text-[length:var(--density-type-caption-lg)]" href={action.href}>{action.label} →</a>
          }
          if (action.tone === 'primary') {
            return <a key={action.label} className="ui-link text-[length:var(--density-type-caption-lg)]" href={action.href}>{action.label}</a>
          }
          return (
            <span key={action.label} className="flex items-center gap-1">
              <span className="text-border">·</span>
              <a className="ui-link-muted inline-flex items-center gap-[5px] text-[length:var(--density-type-caption-lg)]" href={action.href}>
                <GitHubIcon className="size-4" /> {action.label}
              </a>
            </span>
          )
        })}
        </div>
      )}
      description={description}
      leadingIcon={leadingIcon ?? <MeshIcon />}
      title={title}
      titleLevel="h1"
    />
  )
}
