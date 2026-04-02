import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Redo2, TriangleAlert, Undo2, X } from 'lucide-react';

import * as DialogPrimitive from '@radix-ui/react-dialog';

import { broadcastConfig, type ConfigValidationError } from '../../lib/api';
import { AUTHORED_CONFIG_VERSION, parseConfig, serializeConfig } from '../../lib/config';
import { cn } from '../../lib/utils';
import type { MeshConfig } from '../../types/config';
import { Button } from '../ui/button';
import { useAsyncError } from './ConfigErrorBoundary';

type SaveStatus =
  | { type: 'idle' }
  | { type: 'saving' }
  | { type: 'success'; saved: number; total: number }
  | { type: 'partial'; saved: number; total: number; failed: string[] }
  | { type: 'error'; error: string };

type NodeDiff = {
  nodeId: string;
  added: string[];
  removed: string[];
};

function computeConfigDiff(currentConfig: MeshConfig, baseConfig: MeshConfig): NodeDiff[] {
  const result: NodeDiff[] = [];
  const currentNodesById = new Map(currentConfig.nodes.map((node) => [node.node_id, node]));
  const baseNodesById = new Map(baseConfig.nodes.map((node) => [node.node_id, node]));
  const allNodeIds = Array.from(
    new Set([
      ...currentConfig.nodes.map((n) => n.node_id),
      ...baseConfig.nodes.map((n) => n.node_id),
    ]),
  ).sort((a, b) => a.localeCompare(b));

  for (const nodeId of allNodeIds) {
    const currentNode = currentNodesById.get(nodeId);
    const baseNode = baseNodesById.get(nodeId);

    const currentModels = new Set((currentNode?.models ?? []).map((m) => m.name));
    const baseModels = new Set((baseNode?.models ?? []).map((m) => m.name));

    const added = [...currentModels].filter((m) => !baseModels.has(m));
    const removed = [...baseModels].filter((m) => !currentModels.has(m));

    if (added.length > 0 || removed.length > 0) {
      result.push({ nodeId, added, removed });
    }
  }

  return result;
}

type Props = {
  config: MeshConfig;
  savedConfig?: MeshConfig;
  isDirty: boolean;
  isConfigValid?: boolean;
  invalidReason?: string | null;
  onSaveSuccess: (savedConfig: MeshConfig) => void;
  onRevert?: () => void;
  onBackendErrors?: (errors: ConfigValidationError[]) => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
};

type ToastVariant = 'success' | 'warning' | 'error';

function ToastItem({
  variant,
  testId,
  children,
}: {
  variant: ToastVariant;
  testId: string;
  children: ReactNode;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const frame = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  const Icon = variant === 'success' ? Check : variant === 'warning' ? TriangleAlert : X;

  return (
    <div
      data-testid={testId}
      className={cn(
        'max-w-sm p-3 rounded-lg bg-card/95 backdrop-blur-sm border shadow-lg text-sm flex items-start gap-2.5 transition-all',
        mounted ? 'opacity-100 translate-y-0 duration-300 ease-out' : 'opacity-0 translate-y-2',
        variant === 'success' && 'border-emerald-500/30',
        variant === 'warning' && 'border-amber-500/30',
        variant === 'error' && 'border-destructive/30',
      )}
    >
      <Icon
        className={cn(
          'h-4 w-4 mt-0.5 shrink-0',
          variant === 'success' && 'text-emerald-400',
          variant === 'warning' && 'text-amber-400',
          variant === 'error' && 'text-destructive',
        )}
      />
      <span className="text-foreground">{children}</span>
    </div>
  );
}

export function SaveConfig({
  config,
  savedConfig,
  isDirty,
  isConfigValid = true,
  invalidReason = null,
  onSaveSuccess,
  onRevert,
  onBackendErrors,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
}: Props) {
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ type: 'idle' });
  const [diffDialogOpen, setDiffDialogOpen] = useState(false);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const throwAsyncError = useAsyncError();

  useEffect(() => {
    return () => {
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    };
  }, []);

  const scheduleDismiss = useCallback(() => {
    if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    dismissTimerRef.current = setTimeout(() => {
      setSaveStatus({ type: 'idle' });
    }, 5000);
  }, []);

  const diffEntries = useMemo(
    () => computeConfigDiff(config, savedConfig ?? { version: AUTHORED_CONFIG_VERSION, nodes: [] }),
    [config, savedConfig],
  );

  const handleSave = useCallback(() => {
    if (!isDirty || !isConfigValid) return;
    setDiffDialogOpen(true);
  }, [isDirty, isConfigValid]);

  const handleConfirmSave = useCallback(async () => {
    setDiffDialogOpen(false);

    try {
      const serialized = serializeConfig(config);
      const savedSnapshot = parseConfig(serialized) ?? config;
      setSaveStatus({ type: 'saving' });

      const result = await broadcastConfig(serialized);

      if (result.ok) {
        setSaveStatus({ type: 'success', saved: result.saved ?? 1, total: result.total ?? 1 });
        onSaveSuccess(savedSnapshot);
        onBackendErrors?.([]);
      } else if (result.saved && result.saved > 0) {
        setSaveStatus({
          type: 'partial',
          saved: result.saved,
          total: result.total ?? result.saved,
          failed: result.failed ?? [],
        });
        onSaveSuccess(savedSnapshot);
        onBackendErrors?.([]);
      } else {
        setSaveStatus({ type: 'error', error: result.error ?? 'Failed to save configuration' });
        onBackendErrors?.(result.errors ?? []);
      }

      scheduleDismiss();
    } catch (error) {
      throwAsyncError(error instanceof Error ? error : new Error(String(error)));
    }
  }, [config, onSaveSuccess, onBackendErrors, scheduleDismiss, throwAsyncError]);

  const isSaving = saveStatus.type === 'saving';
  const saveDisabled = !isDirty || isSaving || !isConfigValid;
  const saveTitle = !isConfigValid ? (invalidReason ?? 'Configuration is invalid') : undefined;
  const saveBtnLabel = isSaving ? 'Saving…' : 'Save Config';

  return (
    <>
      <div className="flex items-center gap-2 flex-wrap">
        {(onUndo ?? onRedo) ? (
          <>
            <Button
              variant="outline"
              size="sm"
              disabled={!canUndo}
              onClick={onUndo}
              data-testid="undo-button"
              aria-label="Undo"
            >
              <Undo2 className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!canRedo}
              onClick={onRedo}
              data-testid="redo-button"
              aria-label="Redo"
            >
              <Redo2 className="h-4 w-4" />
            </Button>
            <div className="w-px h-4 bg-border" />
          </>
        ) : null}
        {onRevert ? (
          <Button
            data-testid="revert-config-btn"
            variant="outline"
            size="sm"
            onClick={onRevert}
            disabled={!isDirty || isSaving}
          >
            Revert
          </Button>
        ) : null}

        <Button
          data-testid="save-config-btn"
          size="sm"
          onClick={handleSave}
          disabled={saveDisabled}
          title={saveTitle}
        >
          {saveBtnLabel}
        </Button>

        <div className="fixed bottom-16 right-4 z-50 flex flex-col gap-2">
          {(saveStatus.type === 'success' ||
            saveStatus.type === 'partial' ||
            saveStatus.type === 'error') && (
            <ToastItem
              variant={
                saveStatus.type === 'success'
                  ? 'success'
                  : saveStatus.type === 'partial'
                    ? 'warning'
                    : 'error'
              }
              testId="save-toast"
            >
              {saveStatus.type === 'success' &&
                `Configuration saved to ${saveStatus.saved}/${saveStatus.total} node${saveStatus.total !== 1 ? 's' : ''}. Restart nodes to apply.`}
              {saveStatus.type === 'partial' &&
                `Saved: ${saveStatus.saved}/${saveStatus.total}. Failed: ${saveStatus.failed.join(', ')}. Restart saved nodes to apply.`}
              {saveStatus.type === 'error' && saveStatus.error}
            </ToastItem>
          )}
        </div>
      </div>

      <DialogPrimitive.Root
        open={diffDialogOpen}
        onOpenChange={(open) => {
          if (!open) setDiffDialogOpen(false);
        }}
      >
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/50 transition-opacity duration-200" />
          <DialogPrimitive.Content
            data-testid="config-diff-dialog"
            className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-xl border bg-card p-6 shadow-lg transition-all duration-200"
          >
            <DialogPrimitive.Title className="mb-2 text-base font-semibold">
              Save Configuration
            </DialogPrimitive.Title>
            <DialogPrimitive.Description asChild>
              <div className="mb-4 space-y-3 text-sm text-muted-foreground">
                {diffEntries.length === 0 ? (
                  <p>No model assignment changes detected.</p>
                ) : (
                  <ul className="space-y-2">
                    {diffEntries.map(({ nodeId, added, removed }) => (
                      <li key={nodeId}>
                        <span className="font-medium text-foreground">{nodeId}</span>
                        <ul className="mt-1 space-y-0.5 pl-4">
                          {added.map((model) => (
                            <li key={`+${model}`} className="text-emerald-400">
                              + {model} added
                            </li>
                          ))}
                          {removed.map((model) => (
                            <li key={`-${model}`} className="text-destructive">
                              − {model} removed
                            </li>
                          ))}
                        </ul>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </DialogPrimitive.Description>
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                data-testid="cancel-save-button"
                onClick={() => setDiffDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                data-testid="confirm-save-button"
                onClick={handleConfirmSave}
              >
                Confirm &amp; Save
              </Button>
            </div>
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
    </>
  );
}
