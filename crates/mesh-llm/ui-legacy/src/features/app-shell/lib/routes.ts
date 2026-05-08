export type TopSection = "dashboard" | "chat" | "playground";

export type AppRoute = {
  section: TopSection;
  chatId: string | null;
};

export function normalizeSection(
  section: TopSection,
  allowPlayground = import.meta.env.DEV,
): TopSection {
  if (!allowPlayground && section === "playground") {
    return "dashboard";
  }
  return section;
}

function normalizeRoute(route: AppRoute, allowPlayground = import.meta.env.DEV): AppRoute {
  const section = normalizeSection(route.section, allowPlayground);
  if (section !== route.section) {
    return { section, chatId: null };
  }
  return route;
}

export function sectionFromPathname(
  pathname: string,
  allowPlayground = import.meta.env.DEV,
): TopSection | null {
  if (pathname === "/dashboard" || pathname === "/dashboard/") {
    return "dashboard";
  }
  if (
    pathname === "/chat" ||
    pathname === "/chat/" ||
    pathname.startsWith("/chat/")
  ) {
    return "chat";
  }
  if (allowPlayground && (pathname === "/playground" || pathname === "/playground/")) {
    return "playground";
  }
  return null;
}

export function readRouteFromLocation(allowPlayground = import.meta.env.DEV): AppRoute {
  if (typeof window === "undefined") {
    return { section: "dashboard", chatId: null };
  }

  const pathname = window.location.pathname;
  if (pathname === "/dashboard" || pathname === "/dashboard/") {
    return { section: "dashboard", chatId: null };
  }
  if (pathname === "/chat" || pathname === "/chat/") {
    return { section: "chat", chatId: null };
  }
  if (pathname.startsWith("/chat/")) {
    const raw = pathname.slice("/chat/".length);
    const chatId = raw ? decodeURIComponent(raw.split("/")[0]) : null;
    return { section: "chat", chatId };
  }
  if (allowPlayground && (pathname === "/playground" || pathname === "/playground/")) {
    return { section: "playground", chatId: null };
  }

  return { section: "dashboard", chatId: null };
}

export function pathnameForRoute(
  route: AppRoute,
  allowPlayground = import.meta.env.DEV,
): string {
  const normalizedRoute = normalizeRoute(route, allowPlayground);

  if (normalizedRoute.section === "dashboard") {
    return "/dashboard";
  }
  if (normalizedRoute.section === "playground") {
    return "/playground";
  }
  return normalizedRoute.chatId ? `/chat/${encodeURIComponent(normalizedRoute.chatId)}` : "/chat";
}

export function pushRoute(route: AppRoute) {
  if (typeof window === "undefined") return;

  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath) {
    window.history.pushState({}, "", nextPath);
  }
}

export function replaceRoute(route: AppRoute) {
  if (typeof window === "undefined") return;

  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath) {
    window.history.replaceState({}, "", nextPath);
  }
}
