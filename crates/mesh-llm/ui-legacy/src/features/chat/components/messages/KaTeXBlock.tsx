import { useEffect, useRef, useState } from "react";
import katex from "katex";

import "katex/dist/katex.min.css";

export function KaTeXBlock({ math, display }: { math: string; display: boolean }) {
  const [rendered, setRendered] = useState(false);
  const blockRef = useRef<HTMLDivElement | null>(null);
  const inlineRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    setRendered(false);
    const container = display ? blockRef.current : inlineRef.current;
    if (!container) return;

    container.replaceChildren();
    try {
      katex.render(math, container, {
        displayMode: display,
        throwOnError: false,
      });
      setRendered(true);
    } catch {
      container.replaceChildren();
    }

    return () => {
      container.replaceChildren();
    };
  }, [math, display]);

  return display ? (
    <>
      <div ref={blockRef} className={rendered ? "my-2 overflow-x-auto" : "hidden"} />
      {!rendered && (
        <div className="my-2 overflow-x-auto text-sm">
          <code>{math}</code>
        </div>
      )}
    </>
  ) : (
    <>
      <span ref={inlineRef} className={rendered ? undefined : "hidden"} />
      {!rendered && <code>{math}</code>}
    </>
  );
}
