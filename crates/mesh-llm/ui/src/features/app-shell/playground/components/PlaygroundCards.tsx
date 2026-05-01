import { useState } from "react";

import { ModelCard } from "../../../dashboard/components/details";
import { Input } from "../../../../components/ui/input";

type ModelStatus = "warm" | "cold" | "loading";
import { ToggleGroup, ToggleGroupItem } from "../../../../components/ui/toggle-group";

const modelDisplayNames: Record<string, string> = {
  "llama-3.2-1b": "Llama 3.2 1B",
  "gemma-2-2b": "Gemma 2 2B",
};

function ModelPreview() {
  const [modelName, setModelName] = useState("llama-3.2-1b");
  const [subtitle, setSubtitle] = useState("");
  const [sizeGb, setSizeGb] = useState(1.2);
  const [nodeCount, setNodeCount] = useState(1);
  const [status, setStatus] = useState<ModelStatus>("warm");
  const [vision, setVision] = useState(false);
  const [reasoning, setReasoning] = useState(false);
  const [moe, setMoe] = useState(false);

  const cardWidth = "w-[380px]";

  return (
    <div className="space-y-2">
      <div className={`${cardWidth} mx-auto`}>
        <ModelCard
          name={modelName}
          displayName={modelDisplayNames[modelName]}
          subtitle={subtitle || undefined}
          sizeGb={sizeGb}
          nodeCount={nodeCount}
          status={status}
          vision={vision}
          reasoning={reasoning}
          moe={moe}
        />
      </div>

      <hr className="my-2" />

      <div className="space-y-3">
<div className="space-y-1.5">
            <label htmlFor="model-name-input" className="text-xs text-muted-foreground font-medium block">
              Model Name
            </label>
            <Input
              id="model-name-input"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="h-8 text-xs"
              placeholder="llama-3.2-1b"
            />
          </div>

          <div className="space-y-1.5">
            <label htmlFor="subtitle-input" className="text-xs text-muted-foreground font-medium block">
              Subtitle (optional)
            </label>
            <Input
              id="subtitle-input"
              value={subtitle}
              onChange={(e) => setSubtitle(e.target.value)}
              className="h-8 text-xs"
              placeholder="Custom subtitle or leave empty for model name"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
          <div className="space-y-1">
            <label htmlFor="model-size" className="text-xs text-muted-foreground font-medium block">
              Size (GB)
            </label>
            <Input
              id="model-size"
              type="number"
              step="0.1"
              value={sizeGb}
              onChange={(e) => setSizeGb(Number(e.target.value))}
              className="h-8 text-xs"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="node-count" className="text-xs text-muted-foreground font-medium block">
              Node Count
            </label>
            <Input
              id="node-count"
              type="number"
              value={nodeCount}
              onChange={(e) => setNodeCount(Math.max(1, Number(e.target.value)))}
              className="h-8 text-xs"
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <span className="text-xs text-muted-foreground font-medium block">Status</span>
          <ToggleGroup
            type="single"
            value={status}
            onValueChange={(v: string) => v && setStatus(v as ModelStatus)}
            className="grid grid-cols-3 w-full"
          >
            <ToggleGroupItem value="warm" className="text-xs h-7">
              Warm
            </ToggleGroupItem>
            <ToggleGroupItem value="cold" className="text-xs h-7">
              Cold
            </ToggleGroupItem>
            <ToggleGroupItem value="loading" className="text-xs h-7">
              Loading
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        <div className="flex flex-wrap gap-2">
          <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
            <input
              type="checkbox"
              checked={vision}
              onChange={(e) => setVision(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border accent-primary"
            />
            Vision 👁
          </label>
          <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
            <input
              type="checkbox"
              checked={reasoning}
              onChange={(e) => setReasoning(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border accent-primary"
            />
            Reasoning 🧠
          </label>
          <label className="flex items-center gap-2 text-xs cursor-pointer select-none">
            <input
              type="checkbox"
              checked={moe}
              onChange={(e) => setMoe(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-border accent-primary"
            />
            MoE 🧩
          </label>
        </div>
      </div>
    </div>
  );
}

export default function PlaygroundCards() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 overflow-auto p-1">
      <ModelPreview />
    </div>
  );
}
