import { type ChangeEvent, type Dispatch, type MutableRefObject, type Ref, type SetStateAction } from "react";
import { File, FileAudio, ImagePlus, Loader2, Paperclip, RotateCcw, Send, Square } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "../../../../components/ui/alert";
import { Button } from "../../../../components/ui/button";
import { Textarea } from "../../../../components/ui/textarea";
import type { ChatAttachment } from "../../lib/chat-types";
import { PendingAttachmentList } from "./PendingAttachmentList";

export function ChatComposer({
  pendingAttachments,
  composerError,
  setComposerError,
  attachmentSendIssue,
  attachmentPreparationMessage,
  input,
  setInput,
  canChat,
  isSending,
  canRegenerate,
  selectedModelAudio,
  selectedModelMultimodal,
  chatInputRef,
  pdfInputRef,
  imageInputRef,
  audioInputRef,
  fileInputRef,
  onFileSelect,
  onImageSelect,
  onAudioSelect,
  onStop,
  onRegenerate,
  onSubmit,
  onRemovePendingAttachment,
  onRetryPendingAttachment,
}: {
  pendingAttachments: ChatAttachment[];
  composerError: string | null;
  setComposerError: Dispatch<SetStateAction<string | null>>;
  attachmentSendIssue: string | null;
  attachmentPreparationMessage: string | null;
  input: string;
  setInput: (value: string) => void;
  canChat: boolean;
  isSending: boolean;
  canRegenerate: boolean;
  selectedModelAudio: boolean;
  selectedModelMultimodal: boolean;
  chatInputRef: MutableRefObject<HTMLTextAreaElement | null>;
  pdfInputRef: MutableRefObject<HTMLInputElement | null>;
  imageInputRef: MutableRefObject<HTMLInputElement | null>;
  audioInputRef: MutableRefObject<HTMLInputElement | null>;
  fileInputRef: MutableRefObject<HTMLInputElement | null>;
  onFileSelect: (event: ChangeEvent<HTMLInputElement>) => void;
  onImageSelect: (event: ChangeEvent<HTMLInputElement>) => void;
  onAudioSelect: (event: ChangeEvent<HTMLInputElement>) => void;
  onStop: () => void;
  onRegenerate: () => void;
  onSubmit: () => void;
  onRemovePendingAttachment: (attachmentId: string) => void;
  onRetryPendingAttachment: (attachmentId: string) => void;
}) {
  return (
    <div className="shrink-0 box-border w-full max-w-full space-y-2 overflow-hidden p-3 md:space-y-3 md:p-4">
      <PendingAttachmentList
        pendingAttachments={pendingAttachments}
        onRemove={onRemovePendingAttachment}
        onRetry={onRetryPendingAttachment}
      />
      {attachmentPreparationMessage && !composerError ? (
        <div className="flex items-center gap-2 rounded-md border bg-muted/50 px-3 py-2 text-sm text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
          <span>{attachmentPreparationMessage}</span>
        </div>
      ) : composerError || attachmentSendIssue ? (
        <Alert variant="destructive" data-testid="composer-error">
          <AlertTitle>Attachment Issue</AlertTitle>
          <AlertDescription>
            {composerError || attachmentSendIssue}
          </AlertDescription>
        </Alert>
      ) : null}
      <Textarea
        ref={chatInputRef as Ref<HTMLTextAreaElement>}
        value={input}
        onChange={(e) => {
          setComposerError(null);
          setInput(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
          if (e.key === "Escape" && isSending) {
            e.preventDefault();
            onStop();
          }
        }}
        data-testid="chat-input"
        rows={2}
        placeholder={canChat ? "Ask me anything..." : "Waiting for a warm model..."}
        disabled={!canChat}
        className="min-h-[56px] md:min-h-[80px] resize-none text-base md:text-sm"
      />
      <div className="flex items-center justify-between gap-2">
        <div className="hidden md:block text-xs text-muted-foreground">
          {isSending
            ? "Esc to stop. Enter to interrupt and send."
            : "Enter to send. Shift+Enter for newline."}
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={pdfInputRef as Ref<HTMLInputElement>}
            type="file"
            accept="application/pdf"
            className="hidden"
            data-testid="chat-pdf-input"
            onChange={onFileSelect}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => pdfInputRef.current?.click()}
            disabled={!canChat || isSending}
            title="Attach PDF"
            aria-label="Attach PDF"
          >
            <File className="h-4 w-4" />
          </Button>
          <input
            ref={imageInputRef as Ref<HTMLInputElement>}
            type="file"
            accept="image/*"
            className="hidden"
            data-testid="chat-image-input"
            onChange={onImageSelect}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => imageInputRef.current?.click()}
            disabled={!canChat || isSending}
            title="Attach image"
            aria-label="Attach image"
          >
            <ImagePlus className="h-4 w-4" />
          </Button>
          {selectedModelAudio ? (
            <>
              <input
                ref={audioInputRef as Ref<HTMLInputElement>}
                type="file"
                accept="audio/*"
                className="hidden"
                data-testid="chat-audio-input"
                onChange={onAudioSelect}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => audioInputRef.current?.click()}
                disabled={!canChat || isSending}
                title="Attach audio"
                aria-label="Attach audio"
              >
                <FileAudio className="h-4 w-4" />
              </Button>
            </>
          ) : null}
          {selectedModelMultimodal ? (
            <>
              <input
                ref={fileInputRef as Ref<HTMLInputElement>}
                type="file"
                className="hidden"
                data-testid="chat-file-input"
                onChange={onFileSelect}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={!canChat || isSending}
                title="Attach file"
                aria-label="Attach file"
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            </>
          ) : null}
          {isSending ? (
            <Button
              type="button"
              variant="destructive"
              onClick={onStop}
              aria-label="Stop generating"
              className="gap-1.5"
            >
              <Square className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Stop</span>
            </Button>
          ) : (
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={onRegenerate}
              disabled={!canRegenerate}
              aria-label="Regenerate"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          )}
          <Button
            onMouseDown={(e) => e.preventDefault()}
            onClick={onSubmit}
            data-testid="chat-send"
            disabled={
              !canChat ||
              (!input.trim() && pendingAttachments.length === 0) ||
              !!attachmentPreparationMessage
            }
          >
            {isSending ? (
              <Send className="mr-2 h-4 w-4 text-muted-foreground" />
            ) : (
              <Send className="mr-2 h-4 w-4" />
            )}
            {isSending ? "Queue" : "Send"}
          </Button>
        </div>
      </div>
    </div>
  );
}
