import { useCallback, useEffect, useState } from "react";

import {
  pushRoute,
  readRouteFromLocation,
  replaceRoute,
  sectionFromPathname,
  type TopSection,
} from "../lib/routes";

export function useAppRouting() {
  const [section, setSection] = useState<TopSection>(
    () => readRouteFromLocation().section,
  );
  const [routedChatId, setRoutedChatId] = useState<string | null>(
    () => readRouteFromLocation().chatId,
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const current = sectionFromPathname(window.location.pathname);
    if (current == null) {
      const route = readRouteFromLocation();
      replaceRoute(route);
      setSection(route.section);
      setRoutedChatId(route.chatId);
    }

    const onPopState = () => {
      const route = readRouteFromLocation();
      setSection(route.section);
      setRoutedChatId(route.chatId);
    };

    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigateToSection = useCallback(
    (next: TopSection, activeConversationId: string | null) => {
      if (next === section) return;
      const nextChatId = next === "chat" ? activeConversationId : null;
      pushRoute({ section: next, chatId: nextChatId });
      setSection(next);
      setRoutedChatId(nextChatId);
    },
    [section],
  );

  const pushChatRoute = useCallback((chatId: string | null) => {
    pushRoute({ section: "chat", chatId });
    setSection("chat");
    setRoutedChatId(chatId);
  }, []);

  const replaceChatRoute = useCallback((chatId: string | null) => {
    replaceRoute({ section: "chat", chatId });
    setSection("chat");
    setRoutedChatId(chatId);
  }, []);

  return {
    section,
    routedChatId,
    navigateToSection,
    pushChatRoute,
    replaceChatRoute,
  };
}
