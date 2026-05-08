import { Badge } from "../../../components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "../../../components/ui/card";
import type { WakeableNode } from "../../app-shell/lib/status-types";
import { StatusPill } from "./details";

const WAKEABLE_NODE_STATE_LABELS = {
  sleeping: "Sleeping",
  waking: "Waking",
} as const;

export function WakeableCapacity({
  wakeableNodes,
}: {
  wakeableNodes?: WakeableNode[];
}) {
  if (!wakeableNodes?.length) return null;

  return (
    <Card data-testid="wakeable-capacity-section">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-sm">Wakeable Capacity</CardTitle>
          <div className="text-xs text-muted-foreground">
            {wakeableNodes.length} node{wakeableNodes.length === 1 ? "" : "s"}
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          Provider-backed capacity that can be woken on demand. These nodes are kept
          separate from the live topology until they rejoin the mesh.
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {wakeableNodes.map((node) => {
            const etaLabel =
              node.wake_eta_secs == null ? null : formatWakeEta(node.wake_eta_secs);
            return (
              <div
                key={node.logical_id}
                className="rounded-md border bg-muted/20 p-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-medium [overflow-wrap:anywhere]">
                      {node.logical_id}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <StatusPill
                        label={WAKEABLE_NODE_STATE_LABELS[node.state]}
                        tone={node.state === "waking" ? "warn" : "cold"}
                        dot
                      />
                      {node.provider ? (
                        <Badge className="h-5 rounded-full px-2 text-[10px] font-medium text-foreground">
                          {node.provider}
                        </Badge>
                      ) : null}
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted-foreground">
                    <div className="font-medium text-foreground">
                      {node.vram_gb.toFixed(1)} GB
                    </div>
                    <div>VRAM</div>
                  </div>
                </div>

                <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2">
                  <div className="space-y-1 sm:col-span-2">
                    <div className="font-medium text-muted-foreground">Models</div>
                    {node.models.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {node.models.map((model) => (
                          <Badge
                            key={model}
                            className="max-w-full rounded-md px-2 py-0.5 text-[11px] text-foreground"
                          >
                            <span className="truncate [overflow-wrap:anywhere]">{model}</span>
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <div className="text-muted-foreground">No advertised models</div>
                    )}
                  </div>

                  {etaLabel ? (
                    <div className="space-y-1">
                      <div className="font-medium text-muted-foreground">ETA</div>
                      <div className="text-foreground">{etaLabel}</div>
                    </div>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function formatWakeEta(seconds: number) {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  if (seconds < 86_400) return `${Math.ceil(seconds / 3600)} hr`;
  return `${Math.ceil(seconds / 86_400)} d`;
}
