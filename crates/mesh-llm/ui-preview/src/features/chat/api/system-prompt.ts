import { useCallback, useEffect, useState } from 'react'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'

function normalizeStoredSystemPrompt(value: string | null): string {
  if (!value || value.trim().length === 0) return ''
  return value
}

export function readStoredChatSystemPrompt(storageKey = APP_STORAGE_KEYS.chatSystemPrompt): string {
  if (typeof window === 'undefined') return ''

  try {
    return normalizeStoredSystemPrompt(window.localStorage.getItem(storageKey))
  } catch {
    return ''
  }
}

export function writeStoredChatSystemPrompt(value: string, storageKey = APP_STORAGE_KEYS.chatSystemPrompt): void {
  if (typeof window === 'undefined') return

  try {
    const normalizedValue = normalizeStoredSystemPrompt(value)
    if (!normalizedValue) {
      window.localStorage.removeItem(storageKey)
      return
    }

    window.localStorage.setItem(storageKey, normalizedValue)
  } catch {
    return
  }
}

export function usePersistentChatSystemPrompt(storageKey = APP_STORAGE_KEYS.chatSystemPrompt) {
  const [systemPrompt, setSystemPromptState] = useState(() => readStoredChatSystemPrompt(storageKey))

  const setSystemPrompt = useCallback(
    (value: string) => {
      const normalizedValue = normalizeStoredSystemPrompt(value)
      setSystemPromptState(normalizedValue)
      writeStoredChatSystemPrompt(normalizedValue, storageKey)
    },
    [storageKey]
  )

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage || event.key !== storageKey) return
      setSystemPromptState(normalizeStoredSystemPrompt(event.newValue))
    }

    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [storageKey])

  return { systemPrompt, setSystemPrompt }
}
