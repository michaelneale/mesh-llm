import { useEffect, useId, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

type MermaidApi = (typeof import("mermaid"))["default"];

let mermaidPromise: Promise<MermaidApi | null> | null = null;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then((module) => module.default).catch(() => null);
  }

  return mermaidPromise;
}

function readResolvedTheme(): "light" | "dark" {
  if (typeof document === "undefined") return "light";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function MermaidBlock({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const renderId = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [error, setError] = useState<string | null>(null);
  const [rendered, setRendered] = useState(false);
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(() =>
    readResolvedTheme(),
  );

  useEffect(() => {
    if (typeof document === "undefined") return;

    const root = document.documentElement;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const updateTheme = () => setResolvedTheme(readResolvedTheme());

    updateTheme();
    const observer = new MutationObserver(updateTheme);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    media.addEventListener("change", updateTheme);

    return () => {
      observer.disconnect();
      media.removeEventListener("change", updateTheme);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const container = containerRef.current;

    setError(null);
    setRendered(false);
    container?.replaceChildren();

    loadMermaid().then(async (mermaid) => {
      if (cancelled || !mermaid) {
        if (!cancelled) setError("Mermaid failed to load");
        return;
      }

      try {
        mermaid.initialize({
          startOnLoad: false,
          theme: resolvedTheme === "dark" ? "dark" : "default",
          securityLevel: "strict",
        });
        const { svg } = await mermaid.render(`mermaid-${renderId}`, code);
        if (cancelled || !container) return;

        const parsed = new DOMParser().parseFromString(svg, "image/svg+xml");
        const root = parsed.documentElement;
        if (
          parsed.querySelector("parsererror") ||
          root.nodeName.toLowerCase() !== "svg"
        ) {
          throw new Error("Render failed");
        }

        container.replaceChildren(document.importNode(root, true));
        setRendered(true);
      } catch (e: unknown) {
        if (!cancelled) {
          container?.replaceChildren();
          setError(e instanceof Error ? e.message : "Render failed");
        }
      }
    });

    return () => {
      cancelled = true;
      container?.replaceChildren();
    };
  }, [code, renderId, resolvedTheme]);

  if (error)
    return (
      <pre className="my-2 rounded-lg border border-border/70 bg-background/80 p-3 text-xs text-muted-foreground">
        <code>{code}</code>
      </pre>
    );
  return (
    <div className="my-2 rounded-lg border border-border/70 bg-background/80 p-3">
      {!rendered ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Rendering diagram…
        </div>
      ) : null}
      <div
        ref={containerRef}
        className={rendered ? "overflow-x-auto [&_svg]:max-w-full" : "hidden"}
      />
    </div>
  );
}
