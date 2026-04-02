import { useEffect, useMemo } from "react";

import { deepEqual } from "../lib/config";
import type { MeshConfig } from "../types/config";

export type UseConfigDirtyParams = {
  config: MeshConfig;
  savedConfig: MeshConfig;
};

export function useConfigDirty({ config, savedConfig }: UseConfigDirtyParams): boolean {
  const isDirty = useMemo(
    () => !deepEqual(config, savedConfig),
    [config, savedConfig],
  );

  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isDirty]);

  return isDirty;
}
