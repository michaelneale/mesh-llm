import { useEffect, useMemo, useState } from "react";
import type { MeshModel, StatusPayload } from "../lib/status-types";

export type { LocalInstance, MeshModel, Ownership, Peer, StatusPayload } from "../lib/status-types";

type ModelsPayload = {
  mesh_models: MeshModel[];
};

function modelCatalogKeyFromStatus(status: StatusPayload | null) {
  if (!status) return "";
  const local = [
    status.model_name,
    ...(status.models ?? []),
    ...(status.available_models ?? []),
    ...(status.requested_models ?? []),
    ...(status.serving_models ?? []),
    ...(status.hosted_models ?? []),
  ].join(",");
  const peers = [...(status.peers ?? [])]
    .map((peer) =>
      [
        peer.id,
        ...(peer.models ?? []),
        ...(peer.available_models ?? []),
        ...(peer.requested_models ?? []),
        ...(peer.serving_models ?? []),
        ...(peer.hosted_models ?? []),
      ].join(","),
    )
    .sort()
    .join("|");
  return `${status.node_id}::${local}::${peers}`;
}

export function useStatusStream() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [modelsPayload, setModelsPayload] = useState<ModelsPayload | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    let statusEvents: EventSource | null = null;
    let reconnectTimer: number | null = null;
    let retryMs = 1000;
    const MAX_RETRY_MS = 15000;
    const reconnectStatusMessage =
      "Trying to reconnect automatically. Live updates will resume shortly.";

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const closeStatusEvents = () => {
      if (!statusEvents) return;
      statusEvents.onopen = null;
      statusEvents.onmessage = null;
      statusEvents.onerror = null;
      statusEvents.close();
      statusEvents = null;
    };

    const loadStatus = () => {
      fetch("/api/status")
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json() as Promise<StatusPayload>;
        })
        .then((data) => {
          if (stop) return;
          setStatus(data);
          setStatusError(null);
        })
        .catch((err: Error) => {
          if (stop) return;
          setStatusError(reconnectStatusMessage);
          console.warn("Failed to fetch /api/status:", err.message);
        });
    };

    const scheduleReconnect = () => {
      if (stop || reconnectTimer !== null) return;
      setStatusError(reconnectStatusMessage);
      console.warn("Connection lost. Reconnecting...");
      closeStatusEvents();
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connectStatusEvents();
        retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
      }, retryMs);
    };

    const connectStatusEvents = () => {
      if (stop || statusEvents) return;

      let source: EventSource;
      try {
        source = new EventSource("/api/events");
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "failed to create EventSource";
        console.warn("Failed to connect status stream:", message);
        scheduleReconnect();
        return;
      }

      statusEvents = source;
      source.onopen = () => {
        if (stop) return;
        retryMs = 1000;
        setStatusError(null);
        loadStatus();
      };
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as StatusPayload;
          setStatus(payload);
          setStatusError(null);
        } catch {
          // ignore malformed status event
        }
      };
      source.onerror = () => {
        if (stop) return;
        scheduleReconnect();
      };
    };

    loadStatus();
    connectStatusEvents();

    return () => {
      stop = true;
      clearReconnectTimer();
      closeStatusEvents();
    };
  }, []);

  const modelCatalogKey = useMemo(() => modelCatalogKeyFromStatus(status), [status]);

  useEffect(() => {
    if (!modelCatalogKey) {
      setModelsPayload(null);
      setModelsLoading(false);
      return;
    }

    const controller = new AbortController();
    let cancelled = false;
    setModelsLoading(true);

    fetch("/api/models", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ModelsPayload>;
      })
      .then((data) => {
        if (cancelled) return;
        setModelsPayload(data);
      })
      .catch((err: Error) => {
        if (cancelled || err.name === "AbortError") return;
        console.warn("Failed to fetch /api/models:", err.message);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [modelCatalogKey]);

  return {
    status,
    statusError,
    meshModels: modelsPayload?.mesh_models ?? [],
    modelsLoading,
  };
}
