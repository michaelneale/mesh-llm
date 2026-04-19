import type { SelfNodeAccentProps } from "../types";

import { color } from "../helpers";
import { SCENE_PALETTES } from "../theme/scene";

export function DarkSelfNodeAccent({ style }: SelfNodeAccentProps) {
  return (
    <div
      className="pointer-events-none absolute h-28 w-28 rounded-full"
      style={{
        ...style,
        background:
          "radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(192,132,252,0.14) 18%, rgba(59,130,246,0.08) 42%, transparent 72%)",
        filter: "blur(10px)",
      }}
    />
  );
}

export function LightSelfNodeAccent({
  style,
  selectedModelMatch,
}: SelfNodeAccentProps) {
  const selfNodeColor = SCENE_PALETTES.light.nodes.self;
  const [lr, lg, lb] = color(selfNodeColor.line, 1).map((v, i) => (i < 3 ? Math.round(v * 255) : v));
  const [fr, fg, fb] = color(selfNodeColor.fill, 1).map((v, i) => (i < 3 ? Math.round(v * 255) : v));
  const accentBorder = `rgba(${lr}, ${lg}, ${lb}, ${selectedModelMatch ? 0.44 : 0.34})`;
  const accentFill = `rgba(${fr}, ${fg}, ${fb}, ${selectedModelMatch ? 0.12 : 0.08})`;
  const accentInset = `rgba(${fr}, ${fg}, ${fb}, ${selectedModelMatch ? 0.18 : 0.12})`;

  return (
    <div className="pointer-events-none absolute" style={style}>
      <div
        className="absolute left-1/2 top-1/2 h-16 w-16 -translate-x-1/2 -translate-y-1/2 rounded-[18px]"
        style={{
          background: `linear-gradient(180deg, rgba(255,255,255,0.98), ${accentFill})`,
          border: `1px solid ${accentBorder}`,
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.86), inset 0 0 0 1px ${accentInset}, 0 12px 28px rgba(15,23,42,0.12)`,
        }}
      />
    </div>
  );
}
