import type { ReactNode } from "react";

import { ExternalLink } from "lucide-react";

export function ModelMetaLinkItem({
  label,
  href,
  text,
  icon,
}: {
  label: string;
  href: string;
  text: string;
  icon?: ReactNode;
}) {
  return (
    <div className="rounded-lg border bg-muted/25 px-3 py-2">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
        {icon ? <span className="shrink-0">{icon}</span> : null}
        <span>{label}</span>
      </div>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-1 inline-flex items-center gap-1.5 text-sm font-medium underline-offset-4 hover:text-foreground hover:underline [overflow-wrap:anywhere]"
      >
        <span>{text}</span>
        <ExternalLink className="h-3 w-3 shrink-0" />
      </a>
    </div>
  );
}
