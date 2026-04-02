import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';

import type { OwnedNode } from '../../../hooks/useOwnedNodes';
import { parseConfig, serializeConfig } from '../../../lib/config';
import type { AggregatedModel } from '../../../lib/models';
import type { MeshConfig } from '../../../types/config';
import {
  ConfigPage,
  getAssignmentId,
  getSplitGroupId,
  moveSplitAssignmentToNode,
  removeAssignmentFromConfig,
  recombineSplitGroupInConfig,
  resizeSplitBoundaryInConfig,
} from '../../../pages/ConfigPage';

let previousDirtySnapshot: MeshConfig | null = null;
let ownedNodesValue: OwnedNode[] = [];
const aggregateModelsMock = vi.fn<() => AggregatedModel[]>(() => []);
const modelCatalogMock = vi.fn(({ peers }: { peers: Array<{ models: string[] }> }) => (
  <div data-testid="model-catalog">{JSON.stringify(peers)}</div>
));

function makeOwnedNode(overrides: Partial<OwnedNode> = {}): OwnedNode {
  return {
    id: 'node-a',
    hostname: 'alpha.local',
    role: 'Host',
    hardwareLabel: 'GPU',
    hardwareNames: [],
    gpuName: 'RTX 4090',
    vramGb: 24,
    models: [],
    statusLabel: 'Serving',
    statusTone: 'serving',
    isSelf: true,
    gpuTargets: [],
    aggregateVramGb: 24,
    separateCapable: false,
    mixedGpuWarning: false,
    ...overrides,
  } as OwnedNode;
}

vi.mock('../../../hooks/useOwnedNodes', () => ({
  useOwnedNodes: vi.fn(() => ownedNodesValue),
}));

vi.mock('../../../lib/models', () => ({
  aggregateModels: () => aggregateModelsMock(),
}));

vi.mock('../ConfigErrorBoundary', () => ({
  ConfigErrorBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('../DndContext', () => ({
  DndContext: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('../EmptyStates', () => ({
  EmptyNoNodes: () => <div data-testid="empty-no-nodes" />,
  EmptyNoSelection: () => <div data-testid="empty-no-selection" />,
  NodeListSkeleton: () => <div data-testid="node-list-skeleton" />,
  CatalogSkeleton: () => <div data-testid="catalog-skeleton" />,
}));

vi.mock('../ModelCatalog', () => ({
  ModelCatalog: (props: { peers: Array<{ models: string[] }> }) => modelCatalogMock(props),
}));

vi.mock('../NodeList', () => ({
  NodeList: () => <div data-testid="node-list" />,
}));

vi.mock('../VramContainer', () => ({
  VramContainer: ({
    nodeId,
    assignments,
    selectedAssignmentIds,
    onSelectAssignment,
    onResizeSplitBoundary,
  }: {
    nodeId: string;
    assignments?: Array<{ id: string; sizeBytes?: number; invalidMessage?: string }>;
    selectedAssignmentIds?: string[];
    onSelectAssignment?: (assignmentId: string) => void;
    onResizeSplitBoundary?: (leftId: string, rightId: string, boundaryStart: number) => void;
  }) => (
    <div>
      <div data-testid={`vram-container-${nodeId}`}>{selectedAssignmentIds?.join(',') ?? ''}</div>
      <div data-testid={`vram-assignments-${nodeId}`}>{JSON.stringify(assignments ?? [])}</div>
      <button
        type="button"
        data-testid={`mock-select-assignment-${nodeId}`}
        onClick={() => onSelectAssignment?.(assignments?.[0]?.id ?? 'missing-assignment')}
      >
        Select assignment
      </button>
      <button
        type="button"
        data-testid="mock-resize-boundary"
        onClick={() => onResizeSplitBoundary?.(assignments?.[0]?.id ?? 'left', assignments?.[1]?.id ?? 'right', 28)}
      >
        Resize split
      </button>
    </div>
  ),
}));

vi.mock('../ModelDetailPanel', () => ({
  ModelDetailPanel: ({
    assignmentId,
    modelName,
    onUpdateModel,
  }: {
    assignmentId?: string | null;
    modelName: string | null;
    onUpdateModel?: (assignmentId: string, updates: Partial<MeshConfig['nodes'][number]['models'][number]>) => void;
  }) => (
    <div data-testid="model-detail-panel">
      {modelName ?? 'none'}
      {assignmentId ? (
        <button
          type="button"
          data-testid="mock-update-ctx"
          onClick={() => onUpdateModel?.(assignmentId, { ctx_size: 8192 })}
        >
          Set 8K ctx
        </button>
      ) : null}
    </div>
  ),
}));

vi.mock('../TomlEditor', () => ({
  TomlEditor: ({
    config,
    onConfigChange,
    onParseErrorChange,
  }: {
    config: MeshConfig;
    onConfigChange: (config: MeshConfig) => void;
    onParseErrorChange?: (error: string | null) => void;
  }) => (
    <div>
      <div data-testid="toml-config">{JSON.stringify(config)}</div>
      <button
        type="button"
        data-testid="toml-set-b"
        onClick={() => onConfigChange({ version: 1, nodes: [{ node_id: 'node-b', models: [] }] })}
      >
        Set B
      </button>
      <button
        type="button"
        data-testid="toml-set-c"
        onClick={() => onConfigChange({ version: 1, nodes: [{ node_id: 'node-c', models: [] }] })}
      >
        Set C
      </button>
      <button type="button" data-testid="toml-set-invalid" onClick={() => onParseErrorChange?.('Invalid TOML')}>
        Set invalid
      </button>
      <button type="button" data-testid="toml-set-valid" onClick={() => onParseErrorChange?.(null)}>
        Set valid
      </button>
    </div>
  ),
}));

vi.mock('../SaveConfig', () => ({
  SaveConfig: ({
    config,
    isDirty,
    isConfigValid,
    invalidReason,
    onSaveSuccess,
  }: {
    config: MeshConfig;
    isDirty: boolean;
    isConfigValid?: boolean;
    invalidReason?: string | null;
    onSaveSuccess: (saved: MeshConfig) => void;
  }) => {
    if (isDirty && previousDirtySnapshot === null) {
      previousDirtySnapshot = config;
    }

    return (
      <div>
        <div data-testid="save-is-dirty">{isDirty ? 'dirty' : 'clean'}</div>
        <div data-testid="save-is-valid">{isConfigValid === false ? 'invalid' : 'valid'}</div>
        <div data-testid="save-invalid-reason">{invalidReason ?? ''}</div>
        <button type="button" data-testid="save-success-current" onClick={() => onSaveSuccess(config)}>
          Save current
        </button>
        <button
          type="button"
          data-testid="save-success-previous"
          onClick={() => onSaveSuccess(previousDirtySnapshot ?? config)}
        >
          Save previous
        </button>
      </div>
    );
  },
}));

describe('ConfigPage canonical authored config state', () => {
  beforeEach(() => {
    previousDirtySnapshot = null;
    ownedNodesValue = [];
    aggregateModelsMock.mockReset();
    aggregateModelsMock.mockReturnValue([]);
    modelCatalogMock.mockClear();
    vi.clearAllMocks();
  });

  it('hydrates authored config from /api/config on load', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\n\n[[nodes]]\nnode_id = "hydrated-node"\nmodels = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('toml-config')).toHaveTextContent('hydrated-node');
    });
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    expect(fetch).toHaveBeenCalledWith('/api/config');
  });

  it('shows a visible load error and keeps the last valid config when /api/config is invalid', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('not valid config'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('config-load-error')).toBeInTheDocument();
    });
    expect(screen.getByTestId('toml-config')).toHaveTextContent('"version":1');
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
  });

  it('tracks dirty state against the last saved snapshot provided by save success callback', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    fireEvent.click(screen.getByTestId('toml-set-b'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    fireEvent.click(screen.getByTestId('toml-set-c'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    fireEvent.click(screen.getByTestId('save-success-previous'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    fireEvent.click(screen.getByTestId('save-success-current'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
  });

  it('uses the responsive configurator layout shell when owned nodes are available', async () => {
    ownedNodesValue = [
      makeOwnedNode({ models: ['Qwen3-30B-A3B-Q4_K_M'] }),
    ];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('config-tools-layout')).toBeInTheDocument();
    });

    expect(screen.getByTestId('config-tools-layout')).toHaveClass('lg:grid-cols-2');
    expect(screen.queryByTestId('node-list')).toBeNull();
    expect(screen.getByTestId('model-catalog')).toBeInTheDocument();
    expect(screen.getByTestId('toml-config')).toBeInTheDocument();
    expect(screen.getByTestId('config-node-sections')).toBeInTheDocument();
    expect(screen.getByTestId('config-node-section-node-a')).toBeInTheDocument();
    expect(screen.getByTestId('vram-container-node-a')).toBeInTheDocument();
  });

  it('shows a read-only prompt when owner fingerprint is untrusted', async () => {
    ownedNodesValue = [];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(
      <ConfigPage
        status={{
          owner_id: null,
          owner_fingerprint: 'future-fingerprint',
          owner_fingerprint_verified: false,
          node_id: 'node-a',
          token: 't',
          node_status: 'idle',
          is_host: false,
          is_client: false,
          llama_ready: false,
          model_name: '(idle)',
          api_port: 3131,
          my_vram_gb: 24,
          model_size_gb: 0,
          my_hostname: 'alpha.local',
          peers: [
            {
              id: 'peer-a',
              role: 'Host',
              models: ['GLM-4.7-Flash-Q4_K_M'],
              vram_gb: 48,
              hostname: 'alpha.local',
              owner_fingerprint: 'future-fingerprint',
              owner_fingerprint_verified: false,
              owner_fingerprint_transitive: true,
              gpus: [],
            },
          ],
          mesh_models: [],
          inflight_requests: 0,
          model_sizes: [],
          model_scans: [],
        } as never}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId('empty-no-nodes')).toBeInTheDocument();
    });

    expect(
      screen.getByText('Configuration is read-only until you claim your nodes'),
    ).toBeInTheDocument();
  });

  it('marks the authored config dirty after a split resize mutation', async () => {
    ownedNodesValue = [
      makeOwnedNode({ models: ['Qwen3'] }),
    ];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\n\n[[nodes.models]]\nname = "Qwen3"\nmodel_key = "abc123"\nsplit = { start = 0, end = 24, total = 48 }\n\n[[nodes.models]]\nname = "Qwen3"\nmodel_key = "abc123"\nsplit = { start = 24, end = 48, total = 48 }\n',
        ),
      }),
    );

    render(
      <ConfigPage
        status={{
          owner_id: 'owner-1',
          node_id: 'node-a',
          token: 't',
          node_status: 'idle',
          is_host: false,
          is_client: false,
          llama_ready: false,
          model_name: '(idle)',
          api_port: 3131,
          my_vram_gb: 24,
          model_size_gb: 0,
          my_hostname: 'alpha.local',
          peers: [],
          mesh_models: [],
          inflight_requests: 0,
          model_sizes: [['Qwen3', 2_000_000_000]],
          model_scans: [
            { name: 'Qwen3', model_key: 'abc123', size_bytes: 2_000_000_000, metadata: { total_offloadable_layers: 48 } },
          ],
        } as never}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    fireEvent.click(screen.getByTestId('mock-resize-boundary'));

    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');
    expect(screen.getByTestId('toml-config')).toHaveTextContent('"end":28');
    expect(screen.getByTestId('toml-config')).toHaveTextContent('"start":28');
  });

  it('marks the config invalid when the TOML editor reports a parse error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('save-is-valid')).toHaveTextContent('valid');
    });

    fireEvent.click(screen.getByTestId('toml-set-invalid'));
    expect(screen.getByTestId('save-is-valid')).toHaveTextContent('invalid');
    expect(screen.getByTestId('save-invalid-reason')).toHaveTextContent('Invalid TOML');

    fireEvent.click(screen.getByTestId('toml-set-valid'));
    expect(screen.getByTestId('save-is-valid')).toHaveTextContent('valid');
  });

  it('does not pass status-only client labels into the model catalog peer list', async () => {
    ownedNodesValue = [makeOwnedNode()];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(
      <ConfigPage
        status={{
          owner_id: 'owner-1',
          node_id: 'node-a',
          token: 't',
          node_status: 'idle',
          is_host: false,
          is_client: true,
          llama_ready: false,
          model_name: '(client)',
          serving_models: [],
          api_port: 3131,
          my_vram_gb: 24,
          model_size_gb: 0,
          my_hostname: 'alpha.local',
          peers: [],
          mesh_models: [],
          inflight_requests: 0,
          model_sizes: [],
          model_scans: [],
        } as never}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId('model-catalog')).toBeInTheDocument();
    });

    const [[catalogProps]] = modelCatalogMock.mock.calls;
    expect(catalogProps?.peers[0]?.models ?? []).toEqual([]);
    expect(screen.getByTestId('model-catalog')).not.toHaveTextContent('(client)');
  });

  it('propagates context-size changes into VRAM assignment sizing and flags overcommit immediately', async () => {
    ownedNodesValue = [makeOwnedNode({ models: ['Qwen3'], vramGb: 24 })];
    aggregateModelsMock.mockReturnValue([
      { name: 'Qwen3', sizeBytes: 23_600_000_000, sizeGb: 23.6, nodeIds: ['node-a'] },
    ]);

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\n\n[[nodes.models]]\nname = "Qwen3"\nctx_size = 4096\n',
        ),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('vram-assignments-node-a')).toHaveTextContent('23935544320');
    });

    fireEvent.click(screen.getByTestId('mock-select-assignment-node-a'));
    fireEvent.click(screen.getByTestId('mock-update-ctx'));

    await waitFor(() => {
      expect(screen.getByTestId('vram-assignments-node-a')).toHaveTextContent('24271088640');
    });
    expect(screen.getByTestId('vram-assignments-node-a')).toHaveTextContent('Exceeds available VRAM by 0.3 GB');
  });

  it('propagates split assignment context updates to every sibling block in the group', async () => {
    ownedNodesValue = [makeOwnedNode({ models: ['Qwen3'], vramGb: 24 })];
    aggregateModelsMock.mockReturnValue([
      { name: 'Qwen3', sizeBytes: 22_000_000_000, sizeGb: 22, nodeIds: ['node-a'] },
    ]);

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\n\n[[nodes.models]]\nname = "Qwen3"\nmodel_key = "abc123"\nctx_size = 4096\nsplit = { start = 0, end = 24, total = 48 }\n\n[[nodes.models]]\nname = "Qwen3"\nmodel_key = "abc123"\nctx_size = 4096\nsplit = { start = 24, end = 48, total = 48 }\n',
        ),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('vram-assignments-node-a')).toHaveTextContent('11167772160');
    });

    fireEvent.click(screen.getByTestId('mock-select-assignment-node-a'));
    fireEvent.click(screen.getByTestId('mock-update-ctx'));

    await waitFor(() => {
      expect(screen.getByTestId('vram-assignments-node-a')).toHaveTextContent('11335544320');
    });

    const configSnapshot = screen.getByTestId('toml-config').textContent ?? '';
    expect(configSnapshot.match(/"ctx_size":8192/g)).toHaveLength(2);
  });
});

describe('ConfigPage split normalization helpers', () => {
  it('resizes a split boundary without creating gaps or overlaps', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 16, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 16, end: 32, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 32, end: 48, total: 48 } },
          ],
        },
      ],
    };

    const result = resizeSplitBoundaryInConfig(config, {
      nodeId: 'node-a',
      leftAssignmentId: 'Qwen3::abc123::0-16-48::pooled',
      rightAssignmentId: 'Qwen3::abc123::16-32-48::pooled',
      boundaryStart: 20,
    });

    expect(result).toBeTruthy();
    if (!result) {
      throw new Error('Expected split resize to succeed');
    }
    expect(result.config.nodes[0]?.models).toEqual([
      { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 20, total: 48 } },
      { name: 'Qwen3', model_key: 'abc123', split: { start: 20, end: 32, total: 48 } },
      { name: 'Qwen3', model_key: 'abc123', split: { start: 32, end: 48, total: 48 } },
    ]);
  });

  it('clamps the first split boundary so the leading edge keeps at least two layers', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 24, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 48, total: 48 } },
          ],
        },
      ],
    };

    const result = resizeSplitBoundaryInConfig(config, {
      nodeId: 'node-a',
      leftAssignmentId: 'Qwen3::abc123::0-24-48::pooled',
      rightAssignmentId: 'Qwen3::abc123::24-48-48::pooled',
      boundaryStart: 1,
    });

    expect(result).toBeTruthy();
    if (!result) {
      throw new Error('Expected split resize to succeed');
    }

    expect(result.config.nodes[0]?.models).toEqual([
      { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 2, total: 48 } },
      { name: 'Qwen3', model_key: 'abc123', split: { start: 2, end: 48, total: 48 } },
    ]);
  });

  it('clamps the last split boundary so the trailing edge keeps at least two layers', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 24, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 48, total: 48 } },
          ],
        },
      ],
    };

    const result = resizeSplitBoundaryInConfig(config, {
      nodeId: 'node-a',
      leftAssignmentId: 'Qwen3::abc123::0-24-48::pooled',
      rightAssignmentId: 'Qwen3::abc123::24-48-48::pooled',
      boundaryStart: 47,
    });

    expect(result).toBeTruthy();
    if (!result) {
      throw new Error('Expected split resize to succeed');
    }

    expect(result.config.nodes[0]?.models).toEqual([
      { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 46, total: 48 } },
      { name: 'Qwen3', model_key: 'abc123', split: { start: 46, end: 48, total: 48 } },
    ]);
  });

  it('rejects an invalid split move and leaves authored config unchanged', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 23, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 47, total: 48 } },
          ],
        },
        {
          node_id: 'node-b',
          models: [],
        },
      ],
    };

    const result = moveSplitAssignmentToNode(config, {
      sourceNodeId: 'node-a',
      targetNodeId: 'node-b',
      assignmentId: 'Qwen3::abc123::0-23-48::pooled',
      advertisedModelsByNode: new Map([
        ['node-a', new Set(['Qwen3'])],
        ['node-b', new Set(['OtherModel'])],
      ]),
      advertisedModelKeysByNodeAndName: new Map([
        ['node-a', new Map([['Qwen3', new Set(['abc123'])]])],
        ['node-b', new Map()],
      ]),
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      throw new Error('expected invalid move');
    }
    expect(result.error).toContain('does not advertise Qwen3');
    expect(result.config).toEqual(config);
  });

  it('rejects a split move when the destination node advertises the model name but not the matching model_key', () => {

    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 23, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 47, total: 48 } },
          ],
        },
        {
          node_id: 'node-b',
          models: [],
        },
      ],
    };

    const result = moveSplitAssignmentToNode(config, {
      sourceNodeId: 'node-a',
      targetNodeId: 'node-b',
      assignmentId: 'Qwen3::abc123::0-23-48::pooled',
      advertisedModelsByNode: new Map([
        ['node-a', new Set(['Qwen3'])],
        ['node-b', new Set(['Qwen3'])],
      ]),
      advertisedModelKeysByNodeAndName: new Map([
        ['node-a', new Map([['Qwen3', new Set(['abc123'])]])],
        ['node-b', new Map([['Qwen3', new Set(['different-key'])]])],
      ]),
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      throw new Error('expected invalid move');
    }
    expect(result.error).toContain('matching scan metadata');
    expect(result.config).toEqual(config);
  });

  it('recombines a complete split group back into a single model on one node', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 24, total: 48 }, ctx_size: 8192 },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 48, total: 48 }, ctx_size: 8192 },
          ],
        },
      ],
    };

    const result = recombineSplitGroupInConfig(config, {
      groupId: 'Qwen3::abc123::48',
      targetNodeId: 'node-a',
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      throw new Error('expected recombine to succeed');
    }
    expect(result.config.nodes[0]?.models).toEqual([
      { name: 'Qwen3', model_key: 'abc123', ctx_size: 8192 },
    ]);
  });

  it('rejects recombine when related split blocks still span multiple nodes', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 23, total: 48 } },
          ],
        },
        {
          node_id: 'node-b',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 47, total: 48 } },
          ],
        },
      ],
    };

    const result = recombineSplitGroupInConfig(config, {
      groupId: 'Qwen3::abc123::48',
      targetNodeId: 'node-a',
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      throw new Error('expected recombine to fail');
    }
    expect(result.error).toContain('Move every split block for Qwen3 onto the same node');
    expect(result.config).toEqual(config);
  });

  it('recombines the remaining split block into a full assignment when one moved split assignment is removed', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 23, total: 48 }, ctx_size: 8192 },
          ],
        },
        {
          node_id: 'node-b',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 24, end: 47, total: 48 }, ctx_size: 8192 },
          ],
        },
      ],
    };

    const result = removeAssignmentFromConfig(config, {
      nodeId: 'node-a',
      assignmentId: 'Qwen3::abc123::0-23-48::pooled',
    });

    expect(result.config.nodes[0]?.models).toEqual([]);
    expect(result.config.nodes[1]?.models).toEqual([
      { name: 'Qwen3', model_key: 'abc123', ctx_size: 8192 },
    ]);
    expect(result.replacementAssignmentId).toBe('Qwen3::pooled');
    expect(result.replacementNodeId).toBe('node-b');
  });

  it('keeps remaining split fragments split when more than one fragment remains after deletion', () => {
    const config: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 0, end: 15, total: 48 } },
          ],
        },
        {
          node_id: 'node-b',
          models: [
            { name: 'Qwen3', model_key: 'abc123', split: { start: 16, end: 31, total: 48 } },
            { name: 'Qwen3', model_key: 'abc123', split: { start: 32, end: 47, total: 48 } },
          ],
        },
      ],
    };

    const result = removeAssignmentFromConfig(config, {
      nodeId: 'node-a',
      assignmentId: 'Qwen3::abc123::0-15-48::pooled',
    });

    expect(result.config.nodes[0]?.models).toEqual([]);
    expect(result.config.nodes[1]?.models).toEqual([
      { name: 'Qwen3', model_key: 'abc123', split: { start: 16, end: 31, total: 48 } },
      { name: 'Qwen3', model_key: 'abc123', split: { start: 32, end: 47, total: 48 } },
    ]);
    expect(result.replacementAssignmentId).toBeNull();
    expect(result.replacementNodeId).toBeNull();
  });
});

describe('ConfigPage schema-v1 placement rehydration', () => {
  beforeEach(() => {
    previousDirtySnapshot = null;
    ownedNodesValue = [];
    aggregateModelsMock.mockReset();
    aggregateModelsMock.mockReturnValue([]);
    modelCatalogMock.mockClear();
    vi.clearAllMocks();
  });

  it('rehydrates a v1 config with placement_mode separate and gpu_index 1 correctly', async () => {
    ownedNodesValue = [makeOwnedNode({ id: 'node-a', hostname: 'alpha.local', separateCapable: true })];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\nplacement_mode = "separate"\n\n[[nodes.models]]\nname = "Qwen3"\ngpu_index = 1\n',
        ),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('toml-config')).toHaveTextContent('"placement_mode":"separate"');
    });

    const configJson = JSON.parse(screen.getByTestId('toml-config').textContent ?? '{}') as MeshConfig;
    const nodeA = configJson.nodes.find((n) => n.node_id === 'node-a');
    expect(nodeA?.placement_mode).toBe('separate');
    expect(nodeA?.models[0]?.gpu_index).toBe(1);
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    expect(screen.getByTestId('node-node-a-mode-separate')).toHaveAttribute('data-state', 'active');
    expect(screen.getByTestId('node-node-a-mode-pooled')).toHaveAttribute('data-state', 'inactive');
  });

});

describe('ConfigPage config schema round-trip', () => {
  it('serializeConfig then parseConfig reproduces the full authored config with splits and model_key', () => {
    const original: MeshConfig = {
      version: 1,
      nodes: [
        {
          node_id: 'node-a',
          hostname: 'alpha.local',
          placement_mode: 'pooled',
          models: [
            {
              name: 'Qwen3-30B-A3B-Q4_K_M',
              model_key: 'mk-abc123',
              split: { start: 0, end: 23, total: 48 },
              ctx_size: 8192,
            },
            {
              name: 'Qwen3-30B-A3B-Q4_K_M',
              model_key: 'mk-abc123',
              split: { start: 24, end: 47, total: 48 },
              ctx_size: 4096,
            },
          ],
        },
        {
          node_id: 'node-b',
          placement_mode: 'pooled',
          models: [
            {
              name: 'GLM-4.7-Flash-Q4_K_M',
              model_key: 'mk-glm47',
              moe_experts: 16,
              ctx_size: 4096,
            },
          ],
        },
      ],
    };

    const serialized = serializeConfig(original);
    const parsed = parseConfig(serialized);
    expect(parsed).toEqual(original);
  });
});

describe('ConfigPage placement mode orchestration', () => {
  beforeEach(() => {
    previousDirtySnapshot = null;
    ownedNodesValue = [];
    aggregateModelsMock.mockReset();
    aggregateModelsMock.mockReturnValue([]);
    modelCatalogMock.mockClear();
    vi.clearAllMocks();
  });

  it('switching to separate applies immediately without confirmation', async () => {
    ownedNodesValue = [
      makeOwnedNode({ id: 'node-a', hostname: 'alpha.local', separateCapable: true }),
      makeOwnedNode({ id: 'node-b', hostname: 'beta.local', isSelf: false }),
    ];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\n\n[[nodes.models]]\nname = "Qwen3"\n\n[[nodes]]\nnode_id = "node-b"\n\n[[nodes.models]]\nname = "GLM"\n',
        ),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('toml-config')).toHaveTextContent('Qwen3');
    });

    fireEvent.mouseDown(screen.getByTestId('node-node-a-mode-separate'), { button: 0 });

    expect(screen.queryByTestId('node-node-a-mode-confirm')).not.toBeInTheDocument();

    await waitFor(() => {
      const configJson = JSON.parse(screen.getByTestId('toml-config').textContent ?? '{}') as MeshConfig;
      const nodeA = configJson.nodes.find((n) => n.node_id === 'node-a');
      expect(nodeA?.placement_mode).toBe('separate');
    });

    const configJson = JSON.parse(screen.getByTestId('toml-config').textContent ?? '{}') as MeshConfig;
    const nodeB = configJson.nodes.find((n) => n.node_id === 'node-b');
    expect(nodeB?.models).toHaveLength(1);
    expect(nodeB?.models[0]?.name).toBe('GLM');
  });

  it('switching to pooled requires confirmation with PCIe warning', async () => {
    ownedNodesValue = [
      makeOwnedNode({ id: 'node-a', hostname: 'alpha.local', separateCapable: true }),
    ];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\nplacement_mode = "separate"\n\n[[nodes.models]]\nname = "Qwen3"\ngpu_index = 0\n',
        ),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('toml-config')).toHaveTextContent('Qwen3');
    });

    fireEvent.mouseDown(screen.getByTestId('node-node-a-mode-pooled'), { button: 0 });

    expect(screen.getByTestId('node-node-a-mode-confirm')).toBeInTheDocument();
    expect(screen.getByText(/slower communication channels/i)).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('node-node-a-mode-confirm'));

    await waitFor(() => {
      expect(screen.queryByTestId('node-node-a-mode-confirm')).not.toBeInTheDocument();
    });

    const configJson = JSON.parse(screen.getByTestId('toml-config').textContent ?? '{}') as MeshConfig;
    const nodeA = configJson.nodes.find((n) => n.node_id === 'node-a');
    expect(nodeA?.placement_mode).toBe('pooled');
    expect(nodeA?.models).toEqual([]);
  });

  it('split grouping stays stable across GPU target changes', () => {
    const shardA: MeshConfig['nodes'][number]['models'][number] = {
      name: 'Qwen3',
      model_key: 'abc123',
      split: { start: 0, end: 23, total: 48 },
      gpu_index: 0,
    };
    const shardB: MeshConfig['nodes'][number]['models'][number] = {
      name: 'Qwen3',
      model_key: 'abc123',
      split: { start: 24, end: 47, total: 48 },
      gpu_index: 0,
    };

    const idA = getAssignmentId(shardA);
    const idB = getAssignmentId(shardB);
    const groupA = getSplitGroupId(shardA);
    const groupB = getSplitGroupId(shardB);

    expect(idA).toBe('Qwen3::abc123::0-23-48::gpu-0');
    expect(idB).toBe('Qwen3::abc123::24-47-48::gpu-0');
    expect(groupA).toBe('Qwen3::abc123::48');
    expect(groupA).toBe(groupB);

    const shardBOnGpu1 = { ...shardB, gpu_index: 1 };
    const idBOnGpu1 = getAssignmentId(shardBOnGpu1);
    const groupBOnGpu1 = getSplitGroupId(shardBOnGpu1);

    expect(idBOnGpu1).toBe('Qwen3::abc123::24-47-48::gpu-1');
    expect(idBOnGpu1).not.toBe(idB);
    expect(groupBOnGpu1).toBe(groupA);
  });
});

describe('ConfigPage partial failure invariants', () => {
  beforeEach(() => {
    previousDirtySnapshot = null;
    ownedNodesValue = [];
    vi.clearAllMocks();
  });

  it('savedConfig snapshot stays at last-saved value when onSaveSuccess is not called (partial failure)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    fireEvent.click(screen.getByTestId('toml-set-b'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');
  });

  it('savedConfig snapshot advances only when onSaveSuccess fires with the exact saved config', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    fireEvent.click(screen.getByTestId('toml-set-b'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    fireEvent.click(screen.getByTestId('toml-set-c'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    fireEvent.click(screen.getByTestId('save-success-previous'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');

    fireEvent.click(screen.getByTestId('save-success-current'));
    expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
  });

  it('renders ownership notice with destructive variant when tone is warning', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    const mockStatus = {
      node_id: 'test-node-id-12345',
      my_hostname: 'localhost',
      my_vram_gb: 24,
      owner_fingerprint: 'abc123',
      owner_fingerprint_verified: false,
      peers: [],
      model_name: null,
      serving_models: [],
    };

    render(<ConfigPage status={mockStatus as any} />);

    await waitFor(() => {
      const alert = screen.getByTestId('config-ownership-notice');
      expect(alert).toBeInTheDocument();
      expect(alert).toHaveClass('border-destructive/50');
    });
  });

  it('renders ownership notice with default variant when tone is info', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    const mockStatus = {
      node_id: 'test-node-id-12345',
      my_hostname: 'localhost',
      my_vram_gb: 24,
      owner_fingerprint: 'abc123',
      owner_fingerprint_verified: true,
      peers: [
        {
          id: 'peer-1-id-12345',
          hostname: 'peer-1',
          owner_fingerprint: 'abc123',
          owner_fingerprint_verified: false,
          owner_fingerprint_transitive: false,
        },
      ],
      model_name: null,
      serving_models: [],
    };

    render(<ConfigPage status={mockStatus as any} />);

    await waitFor(() => {
      const alert = screen.getByTestId('config-ownership-notice');
      expect(alert).toBeInTheDocument();
      expect(alert).not.toHaveClass('border-destructive/50');
      expect(alert).toHaveClass('bg-background');
    });
  });

  it('shows NodeListSkeleton and CatalogSkeleton when isConfigLoading is true', async () => {
    ownedNodesValue = [
      makeOwnedNode({ models: ['Qwen3-30B-A3B-Q4_K_M'] }),
    ];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(() => new Promise(() => {})), // Never resolves to keep isConfigLoading true
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('node-list-skeleton')).toBeInTheDocument();
      expect(screen.getByTestId('catalog-skeleton')).toBeInTheDocument();
    });
  });

  it('hides NodeListSkeleton and CatalogSkeleton when isConfigLoading is false', async () => {
    ownedNodesValue = [
      makeOwnedNode({ models: ['Qwen3-30B-A3B-Q4_K_M'] }),
    ];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.queryByTestId('node-list-skeleton')).not.toBeInTheDocument();
      expect(screen.queryByTestId('catalog-skeleton')).not.toBeInTheDocument();
    });
  });

  it('prevents page unload when isDirty is true', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    // Make a config change to set isDirty=true
    fireEvent.click(screen.getByTestId('toml-set-b'));
    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');
    });

    // Dispatch beforeunload event and verify it's prevented
    const event = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
  });

  it('allows page unload when isDirty is false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
      }),
    );

    render(<ConfigPage status={null} />);

    await waitFor(() => {
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    // Dispatch beforeunload event while clean and verify it's not prevented
    const event = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });

  it('blocks save and shows invalid when config has split gap errors', async () => {
    ownedNodesValue = [makeOwnedNode({ id: 'node-a', models: ['Qwen3'] })];

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: vi.fn().mockResolvedValue(
          'version = 1\n\n[[nodes]]\nnode_id = "node-a"\n\n[[nodes.models]]\nname = "Qwen3"\nmodel_key = "abc123"\nsplit = { start = 0, end = 5, total = 15 }\n\n[[nodes.models]]\nname = "Qwen3"\nmodel_key = "abc123"\nsplit = { start = 10, end = 15, total = 15 }\n',
        ),
      }),
    );

    render(
      <ConfigPage
        status={{
          owner_id: 'owner-1',
          node_id: 'node-a',
          token: 't',
          node_status: 'idle',
          is_host: false,
          is_client: false,
          llama_ready: false,
          model_name: '(idle)',
          api_port: 3131,
          my_vram_gb: 24,
          model_size_gb: 0,
          my_hostname: 'alpha.local',
          peers: [],
          mesh_models: [],
          inflight_requests: 0,
          model_sizes: [['Qwen3', 2_000_000_000]],
          model_scans: [
            { name: 'Qwen3', model_key: 'abc123', size_bytes: 2_000_000_000, metadata: { total_offloadable_layers: 15 } },
          ],
        } as never}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId('save-is-valid')).toHaveTextContent('invalid');
    });
    expect(screen.getByTestId('save-invalid-reason')).toHaveTextContent('Gap in split coverage');
  });

  describe('mergeRuntimeIntoConfig seeding', () => {
    beforeEach(() => {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
          ok: true,
          text: vi.fn().mockResolvedValue('version = 1\nnodes = []\n'),
        }),
      );
    });

    it('seeds config from runtime state on initial load', async () => {
      ownedNodesValue = [makeOwnedNode({ id: 'node-a', models: ['Model1'] })];

      render(<ConfigPage status={null} />);

      await waitFor(() => {
        expect(screen.getByTestId('toml-config')).toHaveTextContent('node-a');
        expect(screen.getByTestId('toml-config')).toHaveTextContent('Model1');
      });
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    it('re-seeds config when peers change and config is not dirty', async () => {
      ownedNodesValue = [makeOwnedNode({ id: 'node-a', models: ['Model1'] })];

      const { rerender } = render(<ConfigPage status={null} />);

      await waitFor(() => {
        expect(screen.getByTestId('toml-config')).toHaveTextContent('node-a');
      });
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');

      ownedNodesValue = [
        makeOwnedNode({ id: 'node-a', models: ['Model1'] }),
        makeOwnedNode({ id: 'node-b', models: ['Model2'] }),
      ];
      rerender(<ConfigPage status={null} />);

      await waitFor(() => {
        expect(screen.getByTestId('toml-config')).toHaveTextContent('node-b');
        expect(screen.getByTestId('toml-config')).toHaveTextContent('Model2');
      });
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
    });

    it('does not re-seed config when peers change and config is dirty', async () => {
      ownedNodesValue = [makeOwnedNode({ id: 'node-a', models: ['Model1'] })];

      const { rerender } = render(<ConfigPage status={null} />);

      await waitFor(() => {
        expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('clean');
      });

      fireEvent.click(screen.getByTestId('toml-set-b'));
      await waitFor(() => {
        expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');
      });
      expect(screen.getByTestId('toml-config')).toHaveTextContent('node-b');

      ownedNodesValue = [
        makeOwnedNode({ id: 'node-a', models: ['Model1'] }),
        makeOwnedNode({ id: 'node-c', models: ['Model2'] }),
      ];
      rerender(<ConfigPage status={null} />);

      await waitFor(() => {
        expect(screen.getByTestId('toml-config')).toHaveTextContent('node-b');
      });
      expect(screen.getByTestId('toml-config')).not.toHaveTextContent('node-c');
      expect(screen.getByTestId('save-is-dirty')).toHaveTextContent('dirty');
    });
  });
});
