import type { ReactNode } from 'react'
import type { LinkItem } from '@/features/app-tabs/types'
import { GitHubIcon } from '@/components/icons/GitHubIcon'

type FooterProps = { version: string; productName?: string; links: LinkItem[]; trailingLink?: LinkItem }

function FooterLink({ href, label, children }: { href: string; label: string; children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-[18px]">
      <span aria-hidden="true">·</span>
      <a className={children ? 'ui-link-muted inline-flex items-center gap-[5px]' : 'ui-link-muted'} href={href}>
        {children}{label}
      </a>
    </span>
  )
}

export function Footer({ version, productName = 'mesh-llm', links, trailingLink }: FooterProps) {
  const versionLabel = version.startsWith('v') ? version : `v${version}`

  return (
    <footer className="flex flex-wrap items-center justify-center gap-x-[18px] gap-y-2 px-[18px] pb-7 pt-5 text-[length:var(--density-type-caption-lg)] text-fg-faint">
      <span className="font-mono">{productName} {versionLabel}</span>
      {links.map((link) => (
        <FooterLink key={link.label} href={link.href} label={link.label} />
      ))}
      {trailingLink && (
        <FooterLink href={trailingLink.href} label={trailingLink.label}>
          <GitHubIcon className="size-3" />
        </FooterLink>
      )}
    </footer>
  )
}
