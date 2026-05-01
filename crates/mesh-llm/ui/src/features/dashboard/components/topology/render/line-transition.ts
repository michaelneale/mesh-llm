type NarrowPairDiffArgs = {
  previousPairKeys: Iterable<string>;
  currentPairKeys: Iterable<string>;
  previousPairRouteSignatures?: ReadonlyMap<string, string>;
  currentPairRouteSignatures?: ReadonlyMap<string, string>;
  addedNodeIds: Iterable<string>;
  removedNodeIds: Iterable<string>;
};

type NarrowPairDiffResult = {
  outgoingPairKeys: Set<string>;
  incomingPairKeys: Set<string>;
};

function parsePairKey(pairKey: string): [string, string] | null {
  const [leftId, rightId, ...rest] = pairKey.split("::");
  if (!leftId || !rightId || rest.length > 0) return null;
  return [leftId, rightId];
}

const _sigCache = new Map<string, string[]>();

function parsePairRouteSignatureNodeIds(signature?: string) {
  if (!signature) return [] as string[];

  const cached = _sigCache.get(signature);
  if (cached) return cached;

  let result: string[];
  try {
    const parsed = JSON.parse(signature) as { blockerIds?: unknown };
    result = Array.isArray(parsed.blockerIds)
      ? parsed.blockerIds.filter((value): value is string => typeof value === "string")
      : [];
  } catch {
    result = [];
  }

  // Cap at 256 entries to bound memory; signatures are small and few
  if (_sigCache.size >= 256) _sigCache.clear();
  _sigCache.set(signature, result);
  return result;
}

function collectPairNodeIds(
  pairKey: string,
  previousPairRouteSignatures?: ReadonlyMap<string, string>,
  currentPairRouteSignatures?: ReadonlyMap<string, string>,
) {
  const endpoints = parsePairKey(pairKey);
  if (!endpoints) return new Set<string>();

  return new Set<string>([
    endpoints[0],
    endpoints[1],
    ...parsePairRouteSignatureNodeIds(previousPairRouteSignatures?.get(pairKey)),
    ...parsePairRouteSignatureNodeIds(currentPairRouteSignatures?.get(pairKey)),
  ]);
}

function setIntersects(left: Iterable<string>, right: Set<string>) {
  for (const value of left) {
    if (right.has(value)) return true;
  }
  return false;
}

export function narrowPairDiffToChangedComponent({
  previousPairKeys,
  currentPairKeys,
  previousPairRouteSignatures,
  currentPairRouteSignatures,
  addedNodeIds,
  removedNodeIds,
}: NarrowPairDiffArgs): NarrowPairDiffResult {
  const previous = new Set(previousPairKeys);
  const current = new Set(currentPairKeys);
  const addedNodeIdSet = new Set(addedNodeIds);
  const outgoingPairKeys = new Set(
    [...previous].filter((pairKey) => !current.has(pairKey)),
  );
  const incomingPairKeys = new Set(
    [...current].filter((pairKey) => !previous.has(pairKey)),
  );

  for (const pairKey of previous) {
    if (!current.has(pairKey)) continue;
    if (previousPairRouteSignatures?.get(pairKey) === currentPairRouteSignatures?.get(pairKey)) {
      continue;
    }
    outgoingPairKeys.add(pairKey);
    incomingPairKeys.add(pairKey);
  }

  if (outgoingPairKeys.size === 0 && incomingPairKeys.size === 0) {
    return { outgoingPairKeys, incomingPairKeys };
  }

  const changedPairKeys = new Set([...outgoingPairKeys, ...incomingPairKeys]);
  const seedNodeIds = new Set([...addedNodeIdSet, ...removedNodeIds]);
  if (seedNodeIds.size === 0) {
    return { outgoingPairKeys, incomingPairKeys };
  }

  if (addedNodeIdSet.size > 0) {
    const localNodeIds = new Set(seedNodeIds);
    for (const pairKey of changedPairKeys) {
      const pairNodeIds = collectPairNodeIds(
        pairKey,
        previousPairRouteSignatures,
        currentPairRouteSignatures,
      );
      if (!setIntersects(pairNodeIds, seedNodeIds)) continue;
      for (const nodeId of pairNodeIds) {
        localNodeIds.add(nodeId);
      }
    }

    return {
      outgoingPairKeys: new Set(
        [...outgoingPairKeys].filter((pairKey) => {
          const endpoints = parsePairKey(pairKey);
          return endpoints
            ? localNodeIds.has(endpoints[0]) && localNodeIds.has(endpoints[1])
            : false;
        }),
      ),
      incomingPairKeys: new Set(
        [...incomingPairKeys].filter((pairKey) => {
          const endpoints = parsePairKey(pairKey);
          return endpoints
            ? localNodeIds.has(endpoints[0]) && localNodeIds.has(endpoints[1])
            : false;
        }),
      ),
    };
  }

  const localNodeIds = new Set(seedNodeIds);
  for (const pairKey of outgoingPairKeys) {
    const pairNodeIds = collectPairNodeIds(
      pairKey,
      previousPairRouteSignatures,
      currentPairRouteSignatures,
    );
    if (!setIntersects(pairNodeIds, seedNodeIds)) continue;
    for (const nodeId of pairNodeIds) {
      localNodeIds.add(nodeId);
    }
  }

  const survivorSeedNodeIds = new Set(localNodeIds);
  for (const pairKey of incomingPairKeys) {
    const pairNodeIds = collectPairNodeIds(
      pairKey,
      previousPairRouteSignatures,
      currentPairRouteSignatures,
    );
    if (!setIntersects(pairNodeIds, survivorSeedNodeIds)) continue;
    for (const nodeId of pairNodeIds) {
      localNodeIds.add(nodeId);
    }
  }

  return {
    outgoingPairKeys: new Set(
      [...outgoingPairKeys].filter((pairKey) => {
        const endpoints = parsePairKey(pairKey);
        return endpoints
          ? localNodeIds.has(endpoints[0]) && localNodeIds.has(endpoints[1])
          : false;
      }),
    ),
    incomingPairKeys: new Set(
      [...incomingPairKeys].filter((pairKey) => {
        const endpoints = parsePairKey(pairKey);
        return endpoints
          ? localNodeIds.has(endpoints[0]) && localNodeIds.has(endpoints[1])
          : false;
      }),
    ),
  };
}
