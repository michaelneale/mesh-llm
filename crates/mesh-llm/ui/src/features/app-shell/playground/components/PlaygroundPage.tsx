import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../../../components/ui/tabs";

type PlaygroundPageProps = Record<string, unknown>;

import PlaygroundBaseUI from "./PlaygroundBaseUI";
import PlaygroundCards from "./PlaygroundCards";

export default function PlaygroundPage(_props: PlaygroundPageProps) {
  if (!import.meta.env.DEV) {
    return null;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-4 md:p-6">
      <div className="mb-4 shrink-0 rounded-lg border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-700 dark:text-yellow-300">
        🔬 Developer Playground — UI testing environment (dev builds only)
      </div>

      <Tabs defaultValue="base-ui" className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <TabsList className="mb-2 w-auto shrink-0" aria-label="Playground tabs">
          <TabsTrigger value="base-ui" id="tab-base-ui" aria-controls="tabpanel-base-ui">
            Base UI
          </TabsTrigger>
          <TabsTrigger value="cards" id="tab-cards" aria-controls="tabpanel-cards">
            Cards
          </TabsTrigger>
        </TabsList>

        <div className="flex min-h-0 flex-1 overflow-hidden">
          <TabsContent
            value="base-ui"
            id="tabpanel-base-ui"
            aria-labelledby="tab-base-ui"
            className="mt-0 h-full flex-1"
          >
            <PlaygroundBaseUI />
          </TabsContent>

          <TabsContent
            value="cards"
            id="tabpanel-cards"
            aria-labelledby="tab-cards"
            className="mt-0 h-full flex-1"
          >
            <PlaygroundCards />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
