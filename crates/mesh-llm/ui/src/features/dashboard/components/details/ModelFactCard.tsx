import type { ReactNode } from "react";

import { Card, CardContent } from "../../../../components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../../components/ui/tooltip";

export function ModelFactCard({
  title,
  value,
  icon,
  tooltip,
}: {
  title: string;
  value: string;
  icon: ReactNode;
  tooltip?: string;
}) {
  const card = (
    <Card>
      <CardContent className="p-3">
        <div className="mb-2 flex items-center gap-2 text-muted-foreground">
          {icon}
          <span className="text-xs">{title}</span>
        </div>
        <div className="flex min-w-0 items-center gap-2 text-sm font-semibold text-foreground">
          <span className="truncate">{value}</span>
        </div>
      </CardContent>
    </Card>
  );
  if (!tooltip) return card;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="block w-full rounded-lg bg-transparent text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {card}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
