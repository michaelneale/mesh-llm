import type { ReactNode } from "react";

import { Badge } from "../../../../components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../../components/ui/tooltip";

import { cn } from "../../../../lib/utils";

export function StatusPill({
  label,
  tone,
  dot = false,
  icon,
  className,
  tooltip,
}: {
  label: string;
  tone: "warm" | "cold" | "good" | "info" | "warn" | "bad" | "neutral";
  dot?: boolean;
  icon?: ReactNode;
  className?: string;
  tooltip?: string;
}) {
  const badge = (
    <Badge
      className={cn(
        "h-5 shrink-0 rounded-full px-2 text-[10px] font-medium",
        statusPillToneClass(tone),
        dot || icon ? "gap-1" : "",
        className,
      )}
    >
      {dot ? <span className="h-1.5 w-1.5 rounded-full bg-current" /> : null}
      {icon}
      {label}
    </Badge>
  );
  if (!tooltip) return badge;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex rounded-full bg-transparent p-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {badge}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function statusPillToneClass(
  tone: "warm" | "cold" | "good" | "info" | "warn" | "bad" | "neutral",
) {
  if (tone === "warm" || tone === "good") {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (tone === "cold" || tone === "info") {
    return "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300";
  }
  if (tone === "warn") {
    return "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  if (tone === "bad") {
    return "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300";
  }
  return "border-border bg-muted text-foreground";
}
