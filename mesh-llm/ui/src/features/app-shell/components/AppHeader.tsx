import { type Dispatch, type SetStateAction, useState } from "react";
import {
  Braces,
  Check,
  ChevronDown,
  Copy,
  Laptop,
  Loader2,
  Moon,
  Sun,
  UserPlus,
  Sparkles,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "../../../components/ui/alert";
import { Badge } from "../../../components/ui/badge";
import { Button } from "../../../components/ui/button";
import { BrandIcon } from "../../../components/brand-icon";
import { MeshLlmWordmark } from "../../../components/mesh-llm-wordmark";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from "../../../components/ui/navigation-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "../../../components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../components/ui/tooltip";
import { cn } from "../../../lib/utils";
import type { TopSection } from "../lib/routes";

const DOCS_URL = "https://docs.anarchai.org";

type ThemeMode = "auto" | "light" | "dark";

type AppHeaderSection = {
  key: TopSection;
  label: string;
};

const PUBLIC_AGENT_LAUNCHERS = ["claude", "goose", "opencode"] as const;
const PRIVATE_AGENT_LAUNCHERS = ["claude", "goose"] as const;

function isPlainLeftClick(event: React.MouseEvent<HTMLAnchorElement>) {
  return (
    event.button === 0 &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey
  );
}

export function AppHeader({
  sections,
  section,
  setSection,
  themeMode,
  setThemeMode,
  statusError,
  inviteWithModelCommand,
  inviteWithModelName,
  inviteClientCommand,
  inviteToken,
  apiDirectUrl,
  isPublicMesh,
}: {
  sections: AppHeaderSection[];
  section: TopSection;
  setSection: (section: TopSection) => void;
  themeMode: ThemeMode;
  setThemeMode: Dispatch<SetStateAction<ThemeMode>>;
  statusError: string | null;
  inviteWithModelCommand: string;
  inviteWithModelName: string;
  inviteClientCommand: string;
  inviteToken: string;
  apiDirectUrl: string;
  isPublicMesh: boolean;
}) {
  const agentLaunchers = isPublicMesh
    ? PUBLIC_AGENT_LAUNCHERS
    : PRIVATE_AGENT_LAUNCHERS;
  const [inviteWithModelCopied, setInviteWithModelCopied] = useState(false);
  const [inviteClientCopied, setInviteClientCopied] = useState(false);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [apiDirectCopied, setApiDirectCopied] = useState(false);
  const [isThemePopoverOpen, setIsThemePopoverOpen] = useState(false);

  async function copyInviteWithModelCommand() {
    if (!inviteWithModelCommand) return;
    try {
      await navigator.clipboard.writeText(inviteWithModelCommand);
      setInviteWithModelCopied(true);
      window.setTimeout(() => setInviteWithModelCopied(false), 1500);
    } catch {
      setInviteWithModelCopied(false);
    }
  }

  async function copyInviteClientCommand() {
    if (!inviteClientCommand) return;
    try {
      await navigator.clipboard.writeText(inviteClientCommand);
      setInviteClientCopied(true);
      window.setTimeout(() => setInviteClientCopied(false), 1500);
    } catch {
      setInviteClientCopied(false);
    }
  }

  async function copyInviteToken() {
    if (!inviteToken) return;
    try {
      await navigator.clipboard.writeText(inviteToken);
      setTokenCopied(true);
      window.setTimeout(() => setTokenCopied(false), 1500);
    } catch {
      setTokenCopied(false);
    }
  }

  async function copyApiDirectUrl() {
    if (!apiDirectUrl) return;
    try {
      await navigator.clipboard.writeText(apiDirectUrl);
      setApiDirectCopied(true);
      window.setTimeout(() => setApiDirectCopied(false), 1500);
    } catch {
      setApiDirectCopied(false);
    }
  }

  function selectThemeMode(mode: ThemeMode) {
    setThemeMode(mode);
    setIsThemePopoverOpen(false);
  }

  return (
    <header className="shrink-0 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-2 px-3 md:h-16 md:gap-4 md:px-4">
        <div className="flex min-w-0 items-center gap-0">
          <div className="flex h-10 w-7 shrink-0 items-center justify-start">
            <BrandIcon className="h-6 w-6 text-primary" />
          </div>
          <div className="hidden min-w-0 sm:block">
            <div className="truncate text-base font-semibold">
              <MeshLlmWordmark />
            </div>
          </div>
        </div>
        <NavigationMenu>
          <NavigationMenuList>
            {sections.map(({ key, label }) => (
              <NavigationMenuItem key={key}>
                <NavigationMenuLink asChild>
                  <a
                    href={key === "chat" ? "/chat" : "/dashboard"}
                    onClick={(event) => {
                      if (!isPlainLeftClick(event)) return;
                      event.preventDefault();
                      setSection(key);
                    }}
                    className={navigationMenuTriggerStyle()}
                    data-active={section === key ? "" : undefined}
                    aria-current={section === key ? "page" : undefined}
                  >
                    {label}
                  </a>
                </NavigationMenuLink>
              </NavigationMenuItem>
            ))}
          </NavigationMenuList>
        </NavigationMenu>
        <div className="ml-auto flex items-center gap-2">
           <Popover>
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    aria-label="API access"
                  >
                    <Braces className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>API</TooltipContent>
            </Tooltip>
            <PopoverContent
              className="w-[calc(100vw-2rem)] max-w-[420px] space-y-3"
              align="end"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Braces className="h-4 w-4 text-muted-foreground" />
                  <span>API Access</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  OpenAI-compatible endpoint — works with any app that speaks
                  the OpenAI API.
                </div>
              </div>
              {apiDirectUrl ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {apiDirectUrl}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy endpoint URL"
                    onClick={() => void copyApiDirectUrl()}
                  >
                    {apiDirectCopied ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">
                    Run mesh-llm locally to get an OpenAI-compatible API on your
                    machine:
                  </div>
                  <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                    <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                      {isPublicMesh
                        ? "mesh-llm --auto"
                        : `mesh-llm --auto --join ${inviteToken || "(token)"}`}
                    </code>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-6 w-6 shrink-0"
                      aria-label="Copy command"
                      onClick={() =>
                        void navigator.clipboard.writeText(
                          isPublicMesh
                            ? "mesh-llm --auto"
                            : `mesh-llm --auto --join ${inviteToken || ""}`,
                        )
                      }
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    This gives you{" "}
                    <code className="text-[0.7rem]">
                      http://127.0.0.1:9337/v1
                    </code>{" "}
                    locally — point any OpenAI-compatible app at it.
                  </div>
                </div>
              )}
              <div className="space-y-2 pt-1">
                <div className="text-xs font-medium">Use with agents</div>
                <div className="space-y-1">
                  {agentLaunchers.map((agent) => {
                    const cmd = isPublicMesh
                      ? `mesh-llm ${agent}`
                      : `mesh-llm ${agent} --join ${inviteToken || "(token)"}`;
                    return (
                      <div
                        key={agent}
                        className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5"
                      >
                        <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                          {cmd}
                        </code>
                        <Button
                          type="button"
                          size="icon"
                          variant="ghost"
                          className="h-6 w-6 shrink-0"
                          aria-label={`Copy ${agent} command`}
                          onClick={() => void navigator.clipboard.writeText(cmd)}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </div>
                    );
                  })}
                </div>
                <div className="text-xs text-muted-foreground">
                  Also works with pi and any OpenAI-compatible client.{" "}
                  <a
                    href={`${DOCS_URL}/#agents`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline hover:text-foreground"
                  >
                    Setup guide →
                  </a>
                </div>
              </div>
              <div className="text-xs text-muted-foreground pt-1">
                Don't have it yet?{" "}
                <a
                  href={`${DOCS_URL}/#install`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Install mesh-llm →
                </a>
              </div>
              <div className="text-xs text-muted-foreground pt-1">
                Agents can gossip too —{" "}
                <code className="text-[0.9em]">
                  mesh-llm blackboard install-skill
                </code>{" "}
                <a
                  href={`${DOCS_URL}/#blackboard`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  →
                </a>
              </div>
            </PopoverContent>
          </Popover>
          <Popover>
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    aria-label="Invite"
                    disabled={!inviteToken}
                  >
                    <UserPlus className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>Invite</TooltipContent>
            </Tooltip>
            <PopoverContent
              className="w-[calc(100vw-2rem)] max-w-[420px] space-y-3"
              align="end"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <UserPlus className="h-4 w-4 text-muted-foreground" />
                  <span>Invite to this mesh</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Invite with a model loaded to add compute, or invite as a
                  client for API-only access.
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs font-medium">
                  <span>Contribute compute</span>
                  <Badge className="h-5 gap-1 border-emerald-500/40 bg-emerald-500/10 px-2 text-[10px] text-emerald-700 dark:text-emerald-300">
                    <Sparkles className="h-3 w-3" />
                    Recommended
                  </Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  Joins and serves the model {inviteWithModelName || "selected model"}
                </div>
              </div>
              {inviteWithModelCommand ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {inviteWithModelCommand}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy model command"
                    onClick={() => void copyInviteWithModelCommand()}
                  >
                    {inviteWithModelCopied ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">
                  No warm model selected yet.
                </div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-xs font-medium">Join as client</div>
                <div className="text-xs text-muted-foreground">
                  Connects for API access without loading a model.
                </div>
              </div>
              {inviteClientCommand ? (
                <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                  <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                    {inviteClientCommand}
                  </code>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Copy client command"
                    onClick={() => void copyInviteClientCommand()}
                  >
                    {inviteClientCopied ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">
                  No invite token available yet.
                </div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-xs font-medium">Invite token</div>
                {inviteToken ? (
                  <div className="flex items-center gap-2 rounded-md border bg-muted/40 px-2 py-1.5">
                    <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs">
                      {inviteToken}
                    </code>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7 shrink-0"
                      aria-label="Copy invite token"
                      onClick={() => void copyInviteToken()}
                    >
                      {tokenCopied ? (
                        <Check className="h-3.5 w-3.5" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    No invite token available yet.
                  </div>
                )}
              </div>
            </PopoverContent>
          </Popover>
          <Popover
            open={isThemePopoverOpen}
            onOpenChange={setIsThemePopoverOpen}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-9 gap-1 px-2"
                    aria-label={`Theme: ${themeMode}`}
                  >
                    {themeMode === "auto" ? (
                      <Laptop className="h-4 w-4" />
                    ) : null}
                    {themeMode === "light" ? <Sun className="h-4 w-4" /> : null}
                    {themeMode === "dark" ? <Moon className="h-4 w-4" /> : null}
                    <ChevronDown className="h-3 w-3 text-muted-foreground" />
                  </Button>
                </PopoverTrigger>
              </TooltipTrigger>
              <TooltipContent>Theme</TooltipContent>
            </Tooltip>
            <PopoverContent className="w-40 space-y-1 p-1" align="end">
              <button
                type="button"
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted",
                  themeMode === "auto" ? "bg-muted" : "",
                )}
                onClick={() => selectThemeMode("auto")}
              >
                <span className="flex items-center gap-2">
                  <Laptop className="h-3.5 w-3.5" />
                  Auto
                </span>
                {themeMode === "auto" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : null}
              </button>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted",
                  themeMode === "light" ? "bg-muted" : "",
                )}
                onClick={() => selectThemeMode("light")}
              >
                <span className="flex items-center gap-2">
                  <Sun className="h-3.5 w-3.5" />
                  Light
                </span>
                {themeMode === "light" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : null}
              </button>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-xs hover:bg-muted",
                  themeMode === "dark" ? "bg-muted" : "",
                )}
                onClick={() => selectThemeMode("dark")}
              >
                <span className="flex items-center gap-2">
                  <Moon className="h-3.5 w-3.5" />
                  Dark
                </span>
                {themeMode === "dark" ? (
                  <Check className="h-3.5 w-3.5" />
                ) : null}
              </button>
            </PopoverContent>
          </Popover>
        </div>
      </div>
      {statusError ? (
        <div className="mx-auto w-full max-w-7xl px-4 pb-3">
          <Alert variant="destructive">
            <Loader2 className="h-4 w-4 animate-spin" />
            <AlertTitle>Connection Interrupted</AlertTitle>
            <AlertDescription>{statusError}</AlertDescription>
          </Alert>
        </div>
      ) : null}
    </header>
  );
}
