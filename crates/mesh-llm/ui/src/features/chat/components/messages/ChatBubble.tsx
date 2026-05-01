import { Bot, ChevronDown, File, FileAudio, Loader2, Sparkles, User } from "lucide-react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../../../../components/ui/accordion";
import { ScrollArea } from "../../../../components/ui/scroll-area";
import { shortName } from "../../../app-shell/lib/status-helpers";
import { cn } from "../../../../lib/utils";
import { messageAttachments } from "../../lib/chat-attachments";
import type { ChatMessage } from "../../lib/chat-types";
import { MarkdownMessage } from "./MarkdownMessage";

export function ChatBubble({
  message,
  reasoningOpen,
  onReasoningToggle,
  streaming,
}: {
  message: ChatMessage;
  reasoningOpen: boolean;
  onReasoningToggle: (open: boolean) => void;
  streaming?: boolean;
}) {
  const isUser = message.role === "user";
  const isThinking = !isUser && message.reasoning && !message.content;
  const hasFinishedThinking = !isUser && message.reasoning && !!message.content;
  const attachments = messageAttachments(message);

  return (
    <div
      className={cn("flex", isUser ? "justify-end" : "justify-start")}
      data-testid={isUser ? "chat-bubble-user" : "chat-bubble-assistant"}
    >
      <div className="w-full min-w-0 max-w-[92%] md:max-w-[82%]">
        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted-foreground">
          {isUser ? (
            <User className="h-3.5 w-3.5" />
          ) : (
            <Bot className="h-3.5 w-3.5" />
          )}
          <span>{isUser ? "You" : "Assistant"}</span>
          {message.model ? <span>· {shortName(message.model)}</span> : null}
        </div>

        {isThinking ? (
          <div className="mb-2 rounded-lg border border-dashed">
            <button
              type="button"
              className="flex w-full items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => onReasoningToggle(!reasoningOpen)}
            >
              <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
              <span>Thinking…</span>
              <ChevronDown
                className={cn(
                  "ml-auto h-3 w-3 transition-transform",
                  reasoningOpen ? "" : "-rotate-90",
                )}
              />
            </button>
            {reasoningOpen && message.reasoning ? (
              <div className="border-t border-dashed px-3 pb-2 pt-1">
                <ScrollArea className="max-h-60">
                  <div className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">
                    {message.reasoning}
                  </div>
                </ScrollArea>
              </div>
            ) : null}
          </div>
        ) : null}

        {hasFinishedThinking ? (
          <Accordion
            type="single"
            collapsible
            value={reasoningOpen ? "reasoning" : ""}
            onValueChange={(v) => onReasoningToggle(v === "reasoning")}
            className="mb-2"
          >
            <AccordionItem
              value="reasoning"
              className="rounded-lg border border-dashed px-3"
            >
              <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
                <span className="flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3" />
                  Thought for a moment
                </span>
              </AccordionTrigger>
              <AccordionContent>
                <div className="whitespace-pre-wrap text-xs leading-5 text-muted-foreground">
                  {message.reasoning}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}

        {isUser && attachments.length > 0 ? (
          <div className="mb-2 space-y-2">
            {attachments.map((attachment) =>
              attachment.kind === "image" ? (
                <div key={attachment.id}>
                  <img
                    src={attachment.dataUrl}
                    alt={attachment.fileName || "Attached image"}
                    className="max-h-48 rounded-lg border object-contain"
                  />
                </div>
              ) : attachment.kind === "audio" ? (
                <div
                  key={attachment.id}
                  className="rounded-lg border bg-muted/40 p-3"
                >
                  <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
                    <FileAudio className="h-3.5 w-3.5" />
                    <span>{attachment.fileName || "Audio attachment"}</span>
                  </div>
                  <audio controls className="w-full" src={attachment.dataUrl}>
                    <track kind="captions" />
                  </audio>
                </div>
              ) : (
                <div
                  key={attachment.id}
                  className="rounded-lg border bg-muted/40 p-3 text-sm"
                >
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <File className="h-3.5 w-3.5" />
                    <span>{attachment.fileName || "File attachment"}</span>
                  </div>
                </div>
              ),
            )}
          </div>
        ) : null}

        {isUser || message.content ? (
          <div
            className={cn(
              "rounded-lg border px-4 py-3 text-sm leading-6 break-words",
              isUser
                ? "bg-muted whitespace-pre-wrap"
                : message.error
                  ? "border-destructive/50 text-destructive"
                  : "bg-background",
            )}
          >
            {message.content ? (
              <MarkdownMessage
                content={message.content}
                streaming={streaming}
              />
            ) : !isUser ? (
              "..."
            ) : (
              ""
            )}
          </div>
        ) : null}

        {message.stats ? (
          <div className="mt-1 px-1 text-xs text-muted-foreground">
            {message.stats}
          </div>
        ) : null}
      </div>
    </div>
  );
}
