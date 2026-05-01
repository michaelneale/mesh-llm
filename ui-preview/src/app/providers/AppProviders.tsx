import type { QueryClient } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { DataModeProvider, type DataMode } from '@/lib/data-mode'
import { FeatureFlagProvider } from '@/lib/feature-flags'
import { I18nProvider } from '@/lib/i18n'
import { QueryProvider } from '@/lib/query/QueryProvider'

export type AppProvidersProps = {
  children: ReactNode
  dataModeStorageKey?: string
  initialDataMode?: DataMode
  locale?: string
  persistDataMode?: boolean
  queryClient?: QueryClient
}

export function AppProviders({
  children,
  dataModeStorageKey,
  initialDataMode,
  locale = 'en-US',
  persistDataMode,
  queryClient,
}: AppProvidersProps) {
  return (
    <QueryProvider client={queryClient}>
      <FeatureFlagProvider>
        <DataModeProvider initialMode={initialDataMode} persist={persistDataMode} storageKey={dataModeStorageKey}>
          <I18nProvider locale={locale}>{children}</I18nProvider>
        </DataModeProvider>
      </FeatureFlagProvider>
    </QueryProvider>
  )
}
