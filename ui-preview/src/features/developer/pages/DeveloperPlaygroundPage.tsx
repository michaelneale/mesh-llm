import { useNavigate, useSearch } from '@tanstack/react-router'
import { Flag, MessageSquare, Network, Palette, PanelTop, SlidersHorizontal, Table2 } from 'lucide-react'
import { useCallback } from 'react'
import { TabPanel } from '@/components/ui/TabPanel'
import { ChatComponentsArea } from '@/features/developer/playground/areas/ChatComponentsArea'
import { ConfigurationControlsArea } from '@/features/developer/playground/areas/ConfigurationControlsArea'
import { DataDisplayArea } from '@/features/developer/playground/areas/DataDisplayArea'
import { FeatureFlagsArea } from '@/features/developer/playground/areas/FeatureFlagsArea'
import { MeshVizPerfArea } from '@/features/developer/playground/areas/MeshVizPerfArea'
import { ShellControlsArea } from '@/features/developer/playground/areas/ShellControlsArea'
import { TokensFoundationsArea } from '@/features/developer/playground/areas/TokensFoundationsArea'
import { type DeveloperPlaygroundTabId } from '@/features/developer/playground/developer-playground-tabs'
import { PLAYGROUND_AREAS } from '@/features/developer/playground/registry'
import { useDeveloperPlaygroundState } from '@/features/developer/playground/useDeveloperPlaygroundState'

export function DeveloperPlaygroundPage() {
  const { tab } = useSearch({ from: '/__playground' })
  const navigate = useNavigate({ from: '/__playground' })

  const handleTabChange = useCallback((nextTab: DeveloperPlaygroundTabId) => {
    void navigate({
      search: (previous) => ({ ...previous, tab: nextTab }),
      replace: true,
    })
  }, [navigate])

  return <DeveloperPlaygroundPageContent activeTab={tab} onTabChange={handleTabChange} />
}

type DeveloperPlaygroundPageContentProps = {
  activeTab: DeveloperPlaygroundTabId
  onTabChange: (tab: DeveloperPlaygroundTabId) => void
}

export function DeveloperPlaygroundPageContent({ activeTab, onTabChange }: DeveloperPlaygroundPageContentProps) {
  const playgroundState = useDeveloperPlaygroundState()

  return (
    <div className="space-y-4">
      <header className="rounded-[var(--radius-lg)] border border-border bg-panel px-4 py-3.5">
        <div>
          <div className="type-label text-fg-faint">Development only</div>
          <h1 className="type-display mt-1 text-foreground">Developer playground</h1>
          <p className="type-body mt-2 text-fg-dim">
            Rework previews by component use, keep the console compact, and drive each surface with editable harness-backed state before anything ships into the real app.
          </p>
        </div>
      </header>

      <TabPanel<DeveloperPlaygroundTabId>
        ariaLabel="Developer playground component groups"
        contentClassName="space-y-4 px-0 pt-4"
        onValueChange={onTabChange}
        tabs={[
          {
            value: 'shell-controls',
            label: 'Shell controls',
            icon: PanelTop,
            description: PLAYGROUND_AREAS[0].description,
            content: <ShellControlsArea state={playgroundState} />,
          },
          {
            value: 'data-display',
            label: 'Data display',
            icon: Table2,
            description: PLAYGROUND_AREAS[1].description,
            content: <DataDisplayArea state={playgroundState} />,
          },
          {
            value: 'chat-components',
            label: 'Chat components',
            icon: MessageSquare,
            description: PLAYGROUND_AREAS[2].description,
            content: <ChatComponentsArea state={playgroundState} />,
          },
          {
            value: 'configuration-controls',
            label: 'Configuration controls',
            icon: SlidersHorizontal,
            description: PLAYGROUND_AREAS[3].description,
            content: <ConfigurationControlsArea state={playgroundState} />,
          },
          {
            value: 'tokens-foundations',
            label: 'Tokens and foundations',
            icon: Palette,
            description: PLAYGROUND_AREAS[4].description,
            content: <TokensFoundationsArea />,
          },
          {
            value: 'feature-flags',
            label: 'Feature flags',
            icon: Flag,
            description: PLAYGROUND_AREAS[5].description,
            content: <FeatureFlagsArea />,
          },
          {
            value: 'meshviz-perf',
            label: 'MeshViz 200',
            icon: Network,
            description: PLAYGROUND_AREAS[6].description,
            content: <MeshVizPerfArea />,
          },
        ]}
        value={activeTab}
      />
    </div>
  )
}
