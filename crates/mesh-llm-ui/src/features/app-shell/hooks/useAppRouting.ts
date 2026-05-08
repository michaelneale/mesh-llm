import { useCallback, useEffect, useState } from 'react'

import {
  normalizeSection,
  pushRoute,
  readRouteFromLocation,
  replaceRoute,
  type TopSection
} from '@/features/app-shell/lib/routes'

export function useAppRouting() {
  const [section, setSection] = useState<TopSection>(() => readRouteFromLocation().section)
  const [routedChatId, setRoutedChatId] = useState<string | null>(() => readRouteFromLocation().chatId)

  useEffect(() => {
    if (typeof window === 'undefined') return

    const updateStateFromLocation = () => {
      const route = readRouteFromLocation()
      setSection(route.section)
      setRoutedChatId(route.chatId)
    }

    updateStateFromLocation()

    const onPopState = () => {
      updateStateFromLocation()
    }

    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  const navigateToSection = useCallback(
    (next: TopSection, activeConversationId: string | null) => {
      const normalizedSection = normalizeSection(next)
      if (normalizedSection === section) return
      const nextChatId = normalizedSection === 'chat' ? activeConversationId : null
      pushRoute({ section: normalizedSection, chatId: nextChatId })
      setSection(normalizedSection)
      setRoutedChatId(nextChatId)
    },
    [section]
  )

  const pushChatRoute = useCallback((chatId: string | null) => {
    pushRoute({ section: 'chat', chatId })
    setSection('chat')
    setRoutedChatId(chatId)
  }, [])

  const replaceChatRoute = useCallback((chatId: string | null) => {
    replaceRoute({ section: 'chat', chatId })
    setSection('chat')
    setRoutedChatId(chatId)
  }, [])

  return {
    section,
    routedChatId,
    navigateToSection,
    pushChatRoute,
    replaceChatRoute
  }
}
