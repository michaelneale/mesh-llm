import { useState } from 'react'
import { Activity, AlertTriangle, Database, Info, MoreHorizontal, ServerOff } from 'lucide-react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { CopyInstructionRow } from '@/components/ui/CopyInstructionRow'
import { DestructiveActionDialog } from '@/components/ui/DestructiveActionDialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/DropdownMenu'
import { EmptyState } from '@/components/ui/EmptyState'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { LiveRefreshPill } from '@/components/ui/LiveRefreshPill'
import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'
import { NativeSelect } from '@/components/ui/NativeSelect'
import { Slider } from '@/components/ui/Slider'
import { Sparkline } from '@/components/ui/Sparkline'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { TextInputDialog } from '@/components/ui/TextInputDialog'
import { Tooltip } from '@/components/ui/tooltip'
import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import { env } from '@/lib/env'
import { PlaygroundPanel } from '@/features/developer/playground/primitives'

export function TokensFoundationsArea() {
  const [selectedPeer, setSelectedPeer] = useState(DASHBOARD_HARNESS.peers[0]?.id ?? '')
  const [contextPreview, setContextPreview] = useState('32768')
  const [textDialogOpen, setTextDialogOpen] = useState(false)
  const [destructiveDialogOpen, setDestructiveDialogOpen] = useState(false)
  const [dialogNote, setDialogNote] = useState('Route this preview through carrack before comparing remote peers.')
  const [savedDialogNote, setSavedDialogNote] = useState(dialogNote)
  const [destructiveConfirmations, setDestructiveConfirmations] = useState(0)
  const tokenExamples = [
    { label: 'API target', value: env.apiUrl },
    { label: 'Peer ID', value: DASHBOARD_HARNESS.peers[0]?.shortId ?? '990232e1c1' },
    {
      label: 'Model',
      value: DASHBOARD_HARNESS.models[0]?.fullId ?? DASHBOARD_HARNESS.models[0]?.name ?? 'Qwen3.6-27B-UD-Q4_K_XL'
    }
  ]

  return (
    <>
      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <PlaygroundPanel
          title="Type scale"
          description="Keep the console hierarchy sharp and compact across dark and light themes."
        >
          <div className="space-y-3">
            <div className="type-display text-foreground">Display signal</div>
            <div className="type-headline text-foreground">Headline signal</div>
            <div className="type-panel-title text-foreground">Panel title signal</div>
            <div className="type-body text-fg-dim">
              Body copy explains state, intent, or next action without competing with machine values.
            </div>
            <div className="type-caption text-fg-faint">
              Caption copy handles secondary detail and small inline guidance.
            </div>
            <div className="type-label text-fg-faint">Label copy handles compact operational metadata.</div>
          </div>
        </PlaygroundPanel>

        <PlaygroundPanel
          title="Control styles"
          description="The shared utility actions stay restrained, bordered, and accent-sparse."
        >
          <div className="flex flex-wrap gap-2">
            <button
              className="ui-control inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
              type="button"
            >
              ui-control
            </button>
            <button
              className="ui-control-primary inline-flex items-center rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
              type="button"
            >
              ui-control-primary
            </button>
            <button
              className="ui-control-ghost inline-flex items-center rounded-[var(--radius)] border border-transparent px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
              type="button"
            >
              ui-control-ghost
            </button>
            <button
              className="ui-control-destructive inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
              type="button"
            >
              ui-control-destructive
            </button>
          </div>
        </PlaygroundPanel>
      </div>

      <PlaygroundPanel
        title="Machine strings"
        description="IDs, endpoints, models, and routes stay in mono so operators can scan them quickly."
      >
        <div className="grid gap-2.5 md:grid-cols-3">
          {tokenExamples.map((example) => (
            <div key={example.label} className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
              <div className="type-label text-fg-faint">{example.label}</div>
              <div className="mt-1 break-all font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                {example.value}
              </div>
            </div>
          ))}
        </div>
      </PlaygroundPanel>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <PlaygroundPanel
          title="Shared form primitives"
          description="Native controls keep labels explicit and focus rings consistent across embedded surfaces."
        >
          <div className="space-y-4">
            <NativeSelect
              ariaLabel="Preview peer"
              name="playground-peer-select"
              value={selectedPeer}
              onValueChange={setSelectedPeer}
              options={DASHBOARD_HARNESS.peers.slice(0, 5).map((peer) => ({ value: peer.id, label: peer.hostname }))}
            />
            <Slider
              ariaLabel="Preview context window"
              formatValue={(value) => `${Math.round(Number(value) / 1024)}k`}
              label="Context window"
              max={131072}
              min={4096}
              name="playground-context-slider"
              onValueChange={setContextPreview}
              step={4096}
              unit="tokens"
              value={contextPreview}
            />
          </div>
        </PlaygroundPanel>

        <PlaygroundPanel
          title="Copy and tooltip primitives"
          description="Copy rows and hover help stay compact, keyboard reachable, and tied to real machine strings."
        >
          <div className="space-y-3">
            <CopyInstructionRow label="API target" value={env.apiUrl} hint="Used by live-mode fetches" />
            <CopyInstructionRow
              label="Join command"
              value={DASHBOARD_HARNESS.connect.runCommand}
              prefix="$"
              noWrapValue
            />
            <Tooltip content="Tooltips annotate controls without replacing visible labels.">
              <button
                className="ui-control inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                type="button"
              >
                Hover or focus for detail
              </button>
            </Tooltip>
          </div>
        </PlaygroundPanel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <PlaygroundPanel
          title="Status primitives"
          description="Badges, refresh indicators, and sparklines expose state with text, shape, and motion-safe signal."
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <StatusBadge dot tone="good">
                Serving
              </StatusBadge>
              <StatusBadge dot tone="warn">
                Warming
              </StatusBadge>
              <StatusBadge dot tone="bad">
                Offline
              </StatusBadge>
              <StatusBadge tone="accent">Local route</StatusBadge>
              <StatusBadge size="caption" tone="muted">
                Unsigned
              </StatusBadge>
            </div>
            <LiveRefreshPill>Live refresh every 5s</LiveRefreshPill>
            <div className="grid gap-2 sm:grid-cols-2">
              <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="type-label text-fg-faint">VRAM pressure</div>
                    <div className="mt-1 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                      57% free
                    </div>
                  </div>
                  <Sparkline ariaLabel="VRAM pressure trend" values={[12, 14, 11, 16, 13, 15, 17, 14, 18]} />
                </div>
              </div>
              <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="type-label text-fg-faint">Inflight requests</div>
                    <div className="mt-1 font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
                      0 active
                    </div>
                  </div>
                  <Sparkline
                    ariaLabel="Inflight request trend"
                    color="var(--color-warn)"
                    pointCount={10}
                    values={[4, 8, 5, 12, 6, 14, 7, 9, 5]}
                  />
                </div>
              </div>
            </div>
          </div>
        </PlaygroundPanel>

        <PlaygroundPanel
          title="Menu and dialog primitives"
          description="Portal-based actions stay keyboard reachable and return operators to the compact control surface."
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button
                    className="ui-control inline-flex items-center gap-2 rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                    type="button"
                  >
                    Node actions
                    <MoreHorizontal className="size-3.5" aria-hidden={true} strokeWidth={1.8} />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start">
                  <DropdownMenuItem>Copy peer ID</DropdownMenuItem>
                  <DropdownMenuItem>Open logs</DropdownMenuItem>
                  <DropdownMenuSeparator className="my-1 h-px bg-border-soft" />
                  <DropdownMenuItem tone="destructive">Stop local preview</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <button
                className="ui-control-primary inline-flex items-center rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                onClick={() => setTextDialogOpen(true)}
                type="button"
              >
                Edit note
              </button>
              <button
                className="ui-control-destructive inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                onClick={() => setDestructiveDialogOpen(true)}
                type="button"
              >
                Clear preview queue
              </button>
            </div>
            <div className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
              <div className="type-label text-fg-faint">Saved note</div>
              <div className="mt-1 text-[length:var(--density-type-caption-lg)] text-foreground">{savedDialogNote}</div>
              <div className="mt-1 text-[length:var(--density-type-caption)] text-fg-faint">
                Destructive confirmations in this session:{' '}
                <span className="font-mono text-foreground">{destructiveConfirmations}</span>
              </div>
            </div>
          </div>
          <TextInputDialog
            description="Edit the operator note shown in this playground panel."
            label="Operator note"
            onOpenChange={setTextDialogOpen}
            onSave={setSavedDialogNote}
            onValueChange={setDialogNote}
            open={textDialogOpen}
            placeholder="Add a compact operational note."
            saveLabel="Save note"
            title="Edit playground note"
            value={dialogNote}
          />
          <DestructiveActionDialog
            description="This preview action only increments the confirmation count, but it uses the same destructive confirmation shell as production actions."
            destructiveLabel="Clear queue"
            onConfirm={() => setDestructiveConfirmations((count) => count + 1)}
            onOpenChange={setDestructiveDialogOpen}
            open={destructiveDialogOpen}
            title="Clear queued preview messages?"
          />
        </PlaygroundPanel>
      </div>

      <PlaygroundPanel
        title="Empty and loading states"
        description="Small reusable state blocks keep disconnected, empty, and loading surfaces visible in one place."
      >
        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <div className="min-h-[260px] rounded-[var(--radius)] border border-border bg-background">
            <EmptyState
              description="No remote workers matched the selected role and status filters. Clear filters or wait for gossip to refresh."
              hint="The icon, title, description, and hint all carry state so color is not the only signal."
              icon={<ServerOff className="size-8" aria-hidden={true} strokeWidth={1.6} />}
              title="No peers visible"
              tone="accent"
            />
          </div>
          <div className="space-y-3 rounded-[var(--radius)] border border-border bg-background p-3">
            <div className="flex items-center gap-2 text-[length:var(--density-type-caption-lg)] font-medium text-foreground">
              <Database className="size-4 text-fg-faint" aria-hidden={true} strokeWidth={1.7} />
              Loading model catalog
            </div>
            <LoadingGhostBlock className="h-12" panelShell shimmer />
            <LoadingGhostBlock className="h-9" shimmer />
            <LoadingGhostBlock className="h-9 w-4/5" />
            <div className="flex items-start gap-2 rounded-[var(--radius)] border border-border-soft px-3 py-2 text-[length:var(--density-type-caption)] text-fg-dim">
              <AlertTriangle className="mt-0.5 size-3.5 text-warn" aria-hidden={true} strokeWidth={1.8} />
              Ghost blocks should describe layout only. The surrounding text explains what is loading.
            </div>
          </div>
        </div>
      </PlaygroundPanel>

      <PlaygroundPanel
        title="Banners and overlays"
        description="Shared status surfaces cover icon framing, alert copy, and unavailable live-data fallbacks."
      >
        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <div className="space-y-4">
            <InfoBanner
              title="Harness data active"
              description="Use editable harness rows before wiring the preview into live mesh services."
              status="Preview"
              leadingIcon={<Activity className="size-4" aria-hidden="true" strokeWidth={1.8} />}
              action={
                <button
                  className="ui-control inline-flex rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium"
                  type="button"
                >
                  Review data
                </button>
              }
            />
            <div className="flex items-center gap-3 rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
              <AccentIconFrame>
                <Info className="size-4" aria-hidden="true" strokeWidth={1.8} />
              </AccentIconFrame>
              <div>
                <div className="type-label text-fg-faint">AccentIconFrame</div>
                <div className="text-[length:var(--density-type-caption-lg)] text-fg-dim">
                  Used for restrained, reusable accent affordances.
                </div>
              </div>
            </div>
          </div>
          <LiveDataUnavailableOverlay
            debugTitle="Preview API offline"
            title="Live preview unavailable"
            debugDescription="This reusable overlay keeps children visible while presenting retry and harness-switch actions."
            productionDescription="The preview is waiting for live data. Retry the connection or continue with harness data."
            onRetry={() => undefined}
            onSwitchToTestData={() => undefined}
          >
            <div className="min-h-[260px] rounded-[var(--radius)] border border-border bg-background p-4">
              <div className="type-label text-fg-faint">Underlying component</div>
              <div className="mt-3 grid gap-2">
                <span className="h-8 rounded-[var(--radius)] bg-panel-strong" />
                <span className="h-8 rounded-[var(--radius)] bg-panel-strong" />
                <span className="h-8 rounded-[var(--radius)] bg-panel-strong" />
              </div>
            </div>
          </LiveDataUnavailableOverlay>
        </div>
      </PlaygroundPanel>
    </>
  )
}
