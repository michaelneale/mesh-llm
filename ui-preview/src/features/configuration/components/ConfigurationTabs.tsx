import type { ReactNode } from 'react'
import { TabPanel, type TabPanelItem } from '@/components/ui/TabPanel'
import { configurationNavigationIconClassName } from '@/features/configuration/components/configuration-navigation-class-names'
import type { ConfigurationTabId } from '@/features/configuration/components/configuration-tab-ids'

export type ConfigurationTabItem = {
  id: ConfigurationTabId
  label: string
  content: ReactNode
  accessory?: TabPanelItem<ConfigurationTabId>['accessory']
  dirty?: boolean
  icon?: TabPanelItem<ConfigurationTabId>['icon']
  renderIcon?: TabPanelItem<ConfigurationTabId>['renderIcon']
}

type ConfigurationTabsProps = {
  value: ConfigurationTabId
  onValueChange: (value: ConfigurationTabId) => void
  tabs: readonly ConfigurationTabItem[]
}

export function ConfigurationTabs({ value, onValueChange, tabs }: ConfigurationTabsProps) {
  const tabPanelItems: TabPanelItem<ConfigurationTabId>[] = tabs.map((tab) => ({
    value: tab.id,
    label: tab.label,
    content: tab.content,
    accessory:
      tab.accessory ??
      (tab.dirty ? (
        <span aria-hidden="true" className="size-[5px] rounded-full bg-warn" title="Unsaved changes" />
      ) : undefined),
    icon: tab.icon,
    renderIcon: tab.renderIcon,
    triggerAttributes: { 'data-tab-dirty': tab.dirty ? 'true' : undefined }
  }))

  return (
    <TabPanel
      ariaLabel="Configuration sections"
      iconClassName={configurationNavigationIconClassName}
      onValueChange={onValueChange}
      tabs={tabPanelItems}
      value={value}
    />
  )
}
