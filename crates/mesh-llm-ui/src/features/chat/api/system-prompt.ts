import { useCallback, useEffect, useState } from 'react'
import { APP_STORAGE_KEYS } from '@/features/app-tabs/data'
import { DEFAULT_SYSTEM_PROMPT } from '@/constants/system-prompt'

const CLEARED_SENTINEL = '\u0000'

function normalizeStoredSystemPrompt(value: string | null): string {
  if (!value || value.trim().length === 0) return ''
  return value
}

/** Resolve the initial system prompt: default when never set, stored value otherwise. */
function resolveInitialSystemPrompt(storageKey = APP_STORAGE_KEYS.chatSystemPrompt): string {
  if (typeof window === 'undefined') return DEFAULT_SYSTEM_PROMPT

  try {
    const raw = window.localStorage.getItem(storageKey)
    // Key missing entirely → first-time user gets the default
    if (raw === null) return DEFAULT_SYSTEM_PROMPT
    // Sentinel marker means user explicitly cleared it
    if (raw === CLEARED_SENTINEL) return ''
    return normalizeStoredSystemPrompt(raw)
  } catch {
    return DEFAULT_SYSTEM_PROMPT
  }
}

export function readStoredChatSystemPrompt(storageKey = APP_STORAGE_KEYS.chatSystemPrompt): string {
  return resolveInitialSystemPrompt(storageKey)
}

export function writeStoredChatSystemPrompt(value: string, storageKey = APP_STORAGE_KEYS.chatSystemPrompt): void {
  if (typeof window === 'undefined') return

  try {
    const normalizedValue = normalizeStoredSystemPrompt(value)
    // Store sentinel instead of removing key — preserves "intentionally cleared" state
    if (!normalizedValue) {
      window.localStorage.setItem(storageKey, CLEARED_SENTINEL)
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
