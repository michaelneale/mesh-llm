import { useCallback, useState, type ReactNode } from "react";

import { Button } from "../../../../components/ui/button";
import { Check, Copy } from "lucide-react";

export function ModelMetaItem({
  label,
  value,
  icon,
  copyValue,
}: {
  label: string;
  value: string;
  icon?: ReactNode;
  copyValue?: string;
}) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    if (!copyValue) return;
    try {
      await navigator.clipboard.writeText(copyValue);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }, [copyValue]);

  return (
    <div className="rounded-lg border bg-muted/25 px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {icon ? <span className="shrink-0">{icon}</span> : null}
          <span>{label}</span>
        </div>
        {copyValue ? (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-6 w-6 rounded-full p-0 text-muted-foreground hover:text-foreground"
            onClick={() => void copy()}
            aria-label={copied ? `${label} copied` : `Copy ${label}`}
            title={copied ? "Copied" : `Copy ${label}`}
          >
            {copied ? (
              <Check className="h-3 w-3" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        ) : null}
      </div>
      <div className="mt-1 text-sm font-medium [overflow-wrap:anywhere]">
        {value}
      </div>
    </div>
  );
}
