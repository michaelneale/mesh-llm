import { File, FileAudio, Loader2, X } from "lucide-react";

import { Button } from "../../../../components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../../components/ui/tooltip";
import type { ChatAttachment } from "../../lib/chat-types";

export function PendingAttachmentList({
  pendingAttachments,
  onRemove,
  onRetry,
}: {
  pendingAttachments: ChatAttachment[];
  onRemove: (attachmentId: string) => void;
  onRetry: (attachmentId: string) => void;
}) {
  if (pendingAttachments.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {pendingAttachments.map((attachment) =>
        attachment.kind === "image" ? (
          <div
            key={attachment.id}
            className="relative inline-block"
            data-testid="pending-attachment"
          >
            <img
              src={attachment.dataUrl}
              alt={attachment.fileName || "Attached image"}
              className="h-20 rounded-md border object-cover"
            />
            <button
              type="button"
              onClick={() => onRemove(attachment.id)}
              className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-destructive text-destructive-foreground text-xs hover:bg-destructive/80"
              aria-label="Remove image"
            >
              <X className="h-3 w-3" />
            </button>
            {attachment.status === "uploading" ? (
              <div className="absolute inset-0 flex items-center justify-center rounded-md bg-background/70 text-xs">
                <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                {attachment.extractionSummary || "Describing…"}
              </div>
            ) : attachment.status === "failed" ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-md bg-background/80 text-xs">
                <span className="px-2 text-center">
                  {attachment.error || "Attachment failed"}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => onRetry(attachment.id)}
                >
                  Retry
                </Button>
              </div>
            ) : null}
            {attachment.extractionSummary && attachment.status !== "uploading" ? (
              <div className="absolute inset-x-0 bottom-0 rounded-b-md bg-background/80 px-1.5 py-0.5 text-[10px] text-muted-foreground truncate">
                {attachment.extractionSummary}
              </div>
            ) : null}
          </div>
        ) : (
          <div
            key={attachment.id}
            className="relative inline-flex max-w-full items-center gap-2 rounded-md border bg-muted/40 px-3 py-2 text-sm"
            data-testid="pending-attachment"
          >
            {attachment.kind === "audio" ? (
              <FileAudio className="h-4 w-4 shrink-0 text-muted-foreground" />
            ) : (
              <File className="h-4 w-4 shrink-0 text-muted-foreground" />
            )}
            <div className="min-w-0 flex flex-col">
              <span className="truncate">
                {attachment.fileName ||
                  (attachment.kind === "audio"
                    ? "Audio attachment"
                    : "File attachment")}
              </span>
              {attachment.extractionSummary ? (
                <span className="text-xs text-muted-foreground truncate">
                  {attachment.extractionSummary}
                </span>
              ) : null}
            </div>
            {attachment.status === "uploading" ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            ) : null}
            {attachment.status === "failed" ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-destructive cursor-help">
                    Failed
                  </span>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs">
                  {attachment.error || "Attachment failed"}
                </TooltipContent>
              </Tooltip>
            ) : null}
            <button
              type="button"
              onClick={() => onRemove(attachment.id)}
              className="flex h-5 w-5 items-center justify-center rounded-full text-muted-foreground hover:bg-muted"
              aria-label={`Remove ${attachment.kind}`}
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ),
      )}
    </div>
  );
}
