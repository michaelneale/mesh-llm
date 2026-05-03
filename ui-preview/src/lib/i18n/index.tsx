/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, type ReactNode } from 'react'

type DictionaryKey = 'app.name' | 'tabs.network' | 'tabs.chat' | 'tabs.configuration'

type I18nContextValue = {
  locale: string
  t: (key: DictionaryKey) => string
}

const messages: Record<DictionaryKey, string> = {
  'app.name': 'meshllm',
  'tabs.network': 'Network',
  'tabs.chat': 'Chat',
  'tabs.configuration': 'Configuration',
}

const I18nContext = createContext<I18nContextValue>({
  locale: 'en-US',
  t: (key) => messages[key],
})

export function I18nProvider({ locale, children }: { locale: string; children: ReactNode }) {
  return <I18nContext.Provider value={{ locale, t: (key) => messages[key] }}>{children}</I18nContext.Provider>
}

export function useI18n() {
  return useContext(I18nContext)
}
