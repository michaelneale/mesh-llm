import type { ReactNode } from "react";

import { ArrowLeft } from "lucide-react";

import { Button } from "../../../../components/ui/button";
import {
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "../../../../components/ui/sheet";
import { cn } from "../../../../lib/utils";

type DetailSidebarHeaderTone = "node" | "model";

export function DetailSidebarHeader({
  tone,
  icon,
  title,
  description,
  onBack,
  children,
}: {
  tone: DetailSidebarHeaderTone;
  icon: ReactNode;
  title: ReactNode;
  description: ReactNode;
  onBack?: () => void;
  children?: ReactNode;
}) {
  return (
    <div
      className={cn(
        "border-b bg-gradient-to-br via-background to-background px-6 pb-3 pt-3",
        tone === "node"
          ? "from-emerald-50 dark:from-emerald-950/20"
          : "from-sky-50 dark:from-sky-950/20",
      )}
    >
      <SheetHeader className="space-y-2 text-left">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border bg-background text-primary shadow-sm">
            {icon}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <SheetTitle className="text-lg font-semibold leading-tight tracking-tight [overflow-wrap:anywhere] sm:text-xl">
                {title}
              </SheetTitle>
              {children}
            </div>
            <SheetDescription className="mt-1.5 text-sm text-muted-foreground [overflow-wrap:anywhere]">
              {description}
            </SheetDescription>
          </div>
          {onBack ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 gap-1.5"
              onClick={onBack}
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          ) : null}
        </div>
      </SheetHeader>
    </div>
  );
}
