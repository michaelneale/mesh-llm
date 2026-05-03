import { useState } from 'react'
import { Activity, Info } from 'lucide-react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { CopyInstructionRow } from '@/components/ui/CopyInstructionRow'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { LiveDataUnavailableOverlay } from '@/components/ui/LiveDataUnavailableOverlay'
import { NativeSelect } from '@/components/ui/NativeSelect'
import { Slider } from '@/components/ui/Slider'
import { Tooltip } from '@/components/ui/Tooltip'
import { DASHBOARD_HARNESS } from '@/features/app-tabs/data'
import { env } from '@/lib/env'
import { PlaygroundPanel } from '../primitives'

export function TokensFoundationsArea() {
  const [selectedPeer, setSelectedPeer] = useState(DASHBOARD_HARNESS.peers[0]?.id ?? '')
  const [contextPreview, setContextPreview] = useState('32768')
  const tokenExamples = [
    { label: 'API target', value: env.apiUrl },
    { label: 'Peer ID', value: DASHBOARD_HARNESS.peers[0]?.shortId ?? '990232e1c1' },
    { label: 'Model', value: DASHBOARD_HARNESS.models[0]?.fullId ?? DASHBOARD_HARNESS.models[0]?.name ?? 'Qwen3.6-27B-UD-Q4_K_XL' },
  ]

  return (
    <>
      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <PlaygroundPanel title="Type scale" description="Keep the console hierarchy sharp and compact across dark and light themes.">
          <div className="space-y-3">
            <div className="type-display text-foreground">Display signal</div>
            <div className="type-headline text-foreground">Headline signal</div>
            <div className="type-panel-title text-foreground">Panel title signal</div>
            <div className="type-body text-fg-dim">Body copy explains state, intent, or next action without competing with machine values.</div>
            <div className="type-caption text-fg-faint">Caption copy handles secondary detail and small inline guidance.</div>
            <div className="type-label text-fg-faint">Label copy handles compact operational metadata.</div>
          </div>
        </PlaygroundPanel>

        <PlaygroundPanel title="Control styles" description="The shared utility actions stay restrained, bordered, and accent-sparse.">
          <div className="flex flex-wrap gap-2">
            <button className="ui-control inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium" type="button">ui-control</button>
            <button className="ui-control-primary inline-flex items-center rounded-[var(--radius)] px-3 py-1.5 text-[length:var(--density-type-control)] font-medium" type="button">ui-control-primary</button>
            <button className="ui-control-ghost inline-flex items-center rounded-[var(--radius)] border border-transparent px-3 py-1.5 text-[length:var(--density-type-control)] font-medium" type="button">ui-control-ghost</button>
            <button className="ui-control-destructive inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium" type="button">ui-control-destructive</button>
          </div>
        </PlaygroundPanel>
      </div>

      <PlaygroundPanel title="Machine strings" description="IDs, endpoints, models, and routes stay in mono so operators can scan them quickly.">
        <div className="grid gap-2.5 md:grid-cols-3">
          {tokenExamples.map((example) => (
            <div key={example.label} className="rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
              <div className="type-label text-fg-faint">{example.label}</div>
              <div className="mt-1 break-all font-mono text-[length:var(--density-type-caption-lg)] text-foreground">{example.value}</div>
            </div>
          ))}
        </div>
      </PlaygroundPanel>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <PlaygroundPanel title="Shared form primitives" description="Native controls keep labels explicit and focus rings consistent across embedded surfaces.">
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

        <PlaygroundPanel title="Copy and tooltip primitives" description="Copy rows and hover help stay compact, keyboard reachable, and tied to real machine strings.">
          <div className="space-y-3">
            <CopyInstructionRow label="API target" value={env.apiUrl} hint="Used by live-mode fetches" />
            <CopyInstructionRow label="Join command" value={DASHBOARD_HARNESS.connect.runCommand} prefix="$" noWrapValue />
            <Tooltip content="Tooltips annotate controls without replacing visible labels.">
              <button className="ui-control inline-flex items-center rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium" type="button">
                Hover or focus for detail
              </button>
            </Tooltip>
          </div>
        </PlaygroundPanel>
      </div>

      <PlaygroundPanel title="Banners and overlays" description="Shared status surfaces cover icon framing, alert copy, and unavailable live-data fallbacks.">
        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <div className="space-y-4">
            <InfoBanner
              title="Harness data active"
              description="Use editable harness rows before wiring the preview into live mesh services."
              status="Preview"
              leadingIcon={<Activity className="size-4" aria-hidden="true" strokeWidth={1.8} />}
              action={<button className="ui-control inline-flex rounded-[var(--radius)] border px-3 py-1.5 text-[length:var(--density-type-control)] font-medium" type="button">Review data</button>}
            />
            <div className="flex items-center gap-3 rounded-[var(--radius)] border border-border bg-background px-3 py-2.5">
              <AccentIconFrame><Info className="size-4" aria-hidden="true" strokeWidth={1.8} /></AccentIconFrame>
              <div>
                <div className="type-label text-fg-faint">AccentIconFrame</div>
                <div className="text-[length:var(--density-type-caption-lg)] text-fg-dim">Used for restrained, reusable accent affordances.</div>
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
