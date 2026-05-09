import type { ReactNode } from "react";

export function DashboardPanelEmpty({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex h-full min-h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-4 text-center md:min-h-[20rem]">
      <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-full border bg-background text-muted-foreground">
        {icon}
      </div>
      <div className="text-sm font-medium">{title}</div>
      <div className="mt-1 max-w-md text-xs text-muted-foreground">
        {description}
      </div>
    </div>
  );
}
