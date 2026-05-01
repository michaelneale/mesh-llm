import { Sparkles } from "lucide-react";
import type { ReactNode } from "react";

import { Badge } from "../../../../components/ui/badge";
import { StatusPill } from "./StatusPill";

export type ModelStatus = "warm" | "cold" | "loading" | string;

interface ModelCardProps {
  name: string;
  displayName?: string;
  subtitle?: ReactNode;
  sizeGb: number;
  nodeCount: number;
  status: ModelStatus;
  vision?: boolean;
  reasoning?: boolean;
  moe?: boolean;
  onClick?: () => void;
}

export function ModelCard({
  name,
  displayName,
  subtitle,
  sizeGb,
  nodeCount,
  status,
  vision,
  reasoning,
  moe,
  onClick,
}: ModelCardProps) {
  const shortName = (nameOrDisplay: string) => nameOrDisplay.split("/").pop() ?? nameOrDisplay;
  const displayText = displayName ? shortName(displayName) : shortName(name);

  return (
    <button
      type="button"
      onClick={onClick}
      className="block w-full rounded-md border p-3 text-left transition-colors hover:border-primary/35 hover:bg-muted/30"
    >
      <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-start">
        <div className="flex h-7 w-7 items-center justify-center rounded-md border bg-muted/40 text-muted-foreground">
          <Sparkles className="h-3.5 w-3.5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm font-medium leading-5 [overflow-wrap:anywhere]">{displayText}</div>
            <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
              {vision && <span role="img" aria-label="Vision">👁</span>}
              {reasoning && <span role="img" aria-label="Reasoning">🧠</span>}
              {moe && <span role="img" aria-label="Mixture of Experts">🧩</span>}
            </div>
          </div>
          {subtitle ? (
            <div className="text-xs leading-4 text-muted-foreground [overflow-wrap:anywhere]">{subtitle}</div>
          ) : (
            <div className="text-xs leading-4 text-muted-foreground [overflow-wrap:anywhere]">{name}</div>
          )}
        </div>
        <StatusPill
          className="self-start"
          label={status === "warm" ? "Warm" : status === "cold" ? "Cold" : status}
          tone={status === "warm" ? "warm" : status === "cold" ? "cold" : "neutral"}
          dot
        />
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>{nodeCount} node{nodeCount === 1 ? "" : "s"}</span>
        <span className="flex items-center gap-2">
          {vision && <span role="img" aria-label="Vision">👁</span>}
          {sizeGb.toFixed(1)} GB
        </span>
      </div>
    </button>
  );
}
