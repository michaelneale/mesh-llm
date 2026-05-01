import { useMemo, useState } from 'react'
import type { ConfigurationDefaultsCategoryId, ConfigurationDefaultsHarnessData, ConfigurationDefaultsValues } from '@/features/app-tabs/types'

function isDefaultsCategory(value: string, data: ConfigurationDefaultsHarnessData): value is ConfigurationDefaultsCategoryId {
  return data.categories.some((category) => category.id === value)
}

export function useDefaultsSettingsState(data: ConfigurationDefaultsHarnessData) {
  const firstCategoryId = data.categories[0]?.id ?? 'runtime'
  const [activeCategoryId, setActiveCategoryId] = useState<ConfigurationDefaultsCategoryId>(firstCategoryId)

  const activeCategory = useMemo(
    () => data.categories.find((category) => category.id === activeCategoryId) ?? data.categories[0],
    [activeCategoryId, data.categories],
  )

  const activeSettings = useMemo(
    () => data.settings.filter((setting) => setting.categoryId === activeCategoryId),
    [activeCategoryId, data.settings],
  )

  return {
    activeCategory,
    activeCategoryId,
    activeSettings,
    setActiveCategoryId: (value: string) => {
      if (isDefaultsCategory(value, data)) setActiveCategoryId(value)
    },
  }
}

export function createDefaultsValues(data: ConfigurationDefaultsHarnessData): ConfigurationDefaultsValues {
  return Object.fromEntries(data.settings.map((setting) => [setting.id, setting.control.value]))
}
