import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { OwnedNode } from '../../../hooks/useOwnedNodes';
import type { GpuTarget } from '../../../lib/hardware';
import type { ModelAssignment } from '../../../types/config';
import type { VramAssignment } from '../VramContainer';
import { ConfigNodeSection } from '../ConfigNodeSection';

vi.mock('../VramContainer', () => ({
  VramContainer: ({
    nodeId,
    nodeHostname,
    totalVramBytes,
    assignments,
    selectedAssignmentIds,
    onSelectAssignment,
  }: {
    nodeId: string;
    nodeHostname?: string;
    totalVramBytes?: number;
    assignments?: Array<{ id: string }>;
    selectedAssignmentIds?: string[];
    onSelectAssignment?: (assignmentId: string) => void;
  }) => (
    <div
      data-testid={`vram-container-${nodeId}`}
      data-hostname={nodeHostname ?? ''}
      data-total-vram={String(totalVramBytes ?? 0)}
    >
      {selectedAssignmentIds?.join(',') ?? ''}
      {(assignments ?? []).map((a) => (
        <span key={a.id} data-testid={`mock-assignment-${a.id}`}>
          {a.id}
        </span>
      ))}
      <button
        type="button"
        data-config-assignment-interactive="true"
        data-testid="mock-vram-model-block"
        onClick={() => onSelectAssignment?.('Qwen3::abc123::0-23-48')}
      >
        Assignment block
      </button>
    </div>
  ),
}));

vi.mock('../ModelDetailPanel', () => ({
  ModelDetailPanel: ({ modelName }: { modelName: string | null }) => (
    <div data-config-assignment-interactive="true" data-testid="model-detail-panel">{modelName ?? 'none'}</div>
  ),
}));

const GPU_4090: GpuTarget = {
  index: 0,
  name: 'RTX 4090',
  vramBytes: 24_000_000_000,
  reservedBytes: 0,
  label: 'GPU 0 · RTX 4090 · 24.0 GB',
};

const GPU_3090: GpuTarget = {
  index: 1,
  name: 'RTX 3090',
  vramBytes: 24_000_000_000,
  reservedBytes: 0,
  label: 'GPU 1 · RTX 3090 · 24.0 GB',
};

function makeNode(overrides: Partial<OwnedNode> = {}): OwnedNode {
  return {
    id: 'node-a',
    hostname: 'alpha.local',
    role: 'Host',
    hardwareLabel: 'GPU',
    hardwareNames: [],
    gpuName: 'RTX 4090',
    vramGb: 24,
    models: ['Qwen3'],
    statusLabel: 'Serving',
    statusTone: 'serving',
    isSelf: true,
    gpuTargets: [GPU_4090],
    aggregateVramGb: 24,
    separateCapable: false,
    mixedGpuWarning: false,
    ...overrides,
  };
}

function makeModelAssignment(overrides: Partial<ModelAssignment> = {}): ModelAssignment {
  return {
    name: 'Qwen3',
    model_key: 'abc123',
    split: { start: 0, end: 23, total: 48 },
    ...overrides,
  };
}

function makeVramAssignment(overrides: Partial<VramAssignment> = {}): VramAssignment {
  return {
    id: 'Qwen3::pooled',
    name: 'Qwen3',
    sizeBytes: 22_000_000_000,
    fullSizeBytes: 22_000_000_000,
    weightsBytes: 20_000_000_000,
    contextBytes: 2_000_000_000,
    sizeGb: 22,
    ...overrides,
  };
}

const baseProps = {
  isSelected: false,
  totalVramBytes: 24_000_000_000,
  assignments: [] as VramAssignment[],
  selectedAssignmentId: null as string | null,
  selectedAssignmentIds: [] as string[],
  selectedAssignment: null as ModelAssignment | null,
  selectedAggregated: null,
  selectedMetadata: null,
  selectedGroupId: null as string | null,
  recombineError: null as string | null,
  availableNodeCount: 2,
  onSelectNode: vi.fn(),
  onClearSelectedAssignment: vi.fn(),
  onRemoveModel: vi.fn(),
  onSelectAssignment: vi.fn(),
  onSplitModel: vi.fn(),
  onRecombineGroup: vi.fn(),
  onResizeSplitBoundary: vi.fn(),
  onUpdateModel: vi.fn(),
};

describe('ConfigNodeSection', () => {
  it('renders node summary and selected label', () => {
    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={null}
        selectedAssignmentIds={[]}
        selectedAssignment={null}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={2}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={vi.fn()}
        onRemoveModel={vi.fn()}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    expect(screen.getByTestId('config-node-section-node-a')).toBeInTheDocument();
    expect(screen.getByText('alpha.local')).toBeInTheDocument();
    expect(screen.getByText('Catalog target')).toBeInTheDocument();
    expect(screen.getByTestId('vram-container-node-a')).toBeInTheDocument();
  });

  it('renders safely when node data uses the older shape without hardware metadata', () => {
    const legacyNode = {
      id: 'node-legacy',
      hostname: 'legacy.local',
      role: 'Host',
      gpuName: 'RTX 4090',
      vramGb: 24,
      models: ['Qwen3'],
      statusLabel: 'Serving',
      statusTone: 'serving',
      isSelf: false,
    } as OwnedNode;

    render(
      <ConfigNodeSection
        node={legacyNode}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={null}
        selectedAssignmentIds={[]}
        selectedAssignment={null}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={1}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={vi.fn()}
        onRemoveModel={vi.fn()}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    expect(screen.getByText('legacy.local')).toBeInTheDocument();
    expect(screen.getByText('GPU · RTX 4090')).toBeInTheDocument();
    expect(screen.getByText('Models · Qwen3')).toBeInTheDocument();
  });

  it('shows recombine controls and forwards recombine action', () => {
    const onRecombineGroup = vi.fn();

    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={'Qwen3::abc123::0-23-48'}
        selectedAssignmentIds={['Qwen3::abc123::0-23-48', 'Qwen3::abc123::24-47-48']}
        selectedAssignment={makeModelAssignment()}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={'Qwen3::abc123::48'}
        recombineError={'Need both split blocks on one node.'}
        availableNodeCount={2}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={vi.fn()}
        onRemoveModel={vi.fn()}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={onRecombineGroup}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    expect(screen.getByText(/Related split blocks stay selected together/)).toBeInTheDocument();
    expect(screen.getByText('Need both split blocks on one node.')).toBeInTheDocument();
    expect(screen.getByTestId('model-detail-panel')).toHaveTextContent('Qwen3');

    fireEvent.click(screen.getByRole('button', { name: 'Recombine split' }));
    expect(onRecombineGroup).toHaveBeenCalledWith('Qwen3::abc123::48');
  });

  it('forwards node selection from the section header', () => {
    const onSelectNode = vi.fn();

    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={null}
        selectedAssignmentIds={[]}
        selectedAssignment={null}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={2}
        onSelectNode={onSelectNode}
        onClearSelectedAssignment={vi.fn()}
        onRemoveModel={vi.fn()}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /alpha.local/i }));
    expect(onSelectNode).toHaveBeenCalledWith('node-a');
  });

  it('clears the selected assignment when clicking away from the block and detail panel', () => {
    const onClearSelectedAssignment = vi.fn();

    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={'Qwen3::abc123::0-23-48'}
        selectedAssignmentIds={['Qwen3::abc123::0-23-48']}
        selectedAssignment={makeModelAssignment()}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={2}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={onClearSelectedAssignment}
        onRemoveModel={vi.fn()}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    fireEvent.pointerDown(document.body);
    expect(onClearSelectedAssignment).toHaveBeenCalledTimes(1);
  });

  it('keeps the current selection when clicking inside the selected model surface', () => {
    const onClearSelectedAssignment = vi.fn();

    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={'Qwen3::abc123::0-23-48'}
        selectedAssignmentIds={['Qwen3::abc123::0-23-48']}
        selectedAssignment={makeModelAssignment()}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={2}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={onClearSelectedAssignment}
        onRemoveModel={vi.fn()}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    fireEvent.pointerDown(screen.getByTestId('mock-vram-model-block'));
    fireEvent.pointerDown(screen.getByTestId('model-detail-panel'));

    expect(onClearSelectedAssignment).not.toHaveBeenCalled();
  });

  it('removes the selected assignment when Delete is pressed', () => {
    const onRemoveModel = vi.fn();

    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={'Qwen3::abc123::0-23-48'}
        selectedAssignmentIds={['Qwen3::abc123::0-23-48']}
        selectedAssignment={makeModelAssignment()}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={2}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={vi.fn()}
        onRemoveModel={onRemoveModel}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    fireEvent.keyDown(document, { key: 'Delete' });
    expect(onRemoveModel).toHaveBeenCalledWith('Qwen3', 'Qwen3::abc123::0-23-48');
  });

  it('removes the selected assignment when Backspace is pressed', () => {
    const onRemoveModel = vi.fn();

    render(
      <ConfigNodeSection
        node={makeNode()}
        isSelected={false}
        totalVramBytes={24e9}
        assignments={[]}
        selectedAssignmentId={'Qwen3::abc123::0-23-48'}
        selectedAssignmentIds={['Qwen3::abc123::0-23-48']}
        selectedAssignment={makeModelAssignment()}
        selectedAggregated={null}
        selectedMetadata={null}
        selectedGroupId={null}
        recombineError={null}
        availableNodeCount={2}
        onSelectNode={vi.fn()}
        onClearSelectedAssignment={vi.fn()}
        onRemoveModel={onRemoveModel}
        onSelectAssignment={vi.fn()}
        onSplitModel={vi.fn()}
        onRecombineGroup={vi.fn()}
        onResizeSplitBoundary={vi.fn()}
        onUpdateModel={vi.fn()}
      />,
    );

    fireEvent.keyDown(document, { key: 'Backspace' });
    expect(onRemoveModel).toHaveBeenCalledWith('Qwen3', 'Qwen3::abc123::0-23-48');
  });

  it('ignores the delete hotkey while typing in an input', () => {
    const onRemoveModel = vi.fn();

    render(
      <>
        <input data-testid="hotkey-input" />
        <ConfigNodeSection
          node={makeNode()}
          isSelected={false}
          totalVramBytes={24e9}
          assignments={[]}
          selectedAssignmentId={'Qwen3::abc123::0-23-48'}
          selectedAssignmentIds={['Qwen3::abc123::0-23-48']}
          selectedAssignment={makeModelAssignment()}
          selectedAggregated={null}
          selectedMetadata={null}
          selectedGroupId={null}
          recombineError={null}
          availableNodeCount={2}
          onSelectNode={vi.fn()}
          onClearSelectedAssignment={vi.fn()}
          onRemoveModel={onRemoveModel}
          onSelectAssignment={vi.fn()}
          onSplitModel={vi.fn()}
          onRecombineGroup={vi.fn()}
          onResizeSplitBoundary={vi.fn()}
          onUpdateModel={vi.fn()}
        />
      </>,
    );

    fireEvent.keyDown(screen.getByTestId('hotkey-input'), { key: 'Delete' });
    expect(onRemoveModel).not.toHaveBeenCalled();
  });

  describe('pooled mode', () => {
    it('renders a pool dropzone wrapping the VramContainer', () => {
      render(
        <ConfigNodeSection {...baseProps} node={makeNode()} placementMode="pooled" />,
      );

      const dropzone = screen.getByTestId('node-node-a-pool-dropzone');
      expect(dropzone).toBeInTheDocument();
      expect(within(dropzone).getByTestId('vram-container-node-a')).toBeInTheDocument();
    });

    it('passes aggregate totalVramBytes to the single container', () => {
      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode()}
          placementMode="pooled"
          totalVramBytes={48_000_000_000}
        />,
      );

      const container = screen.getByTestId('vram-container-node-a');
      expect(container).toHaveAttribute('data-total-vram', '48000000000');
    });

    it('renders exactly one VramContainer even with multiple GPU targets', () => {
      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, GPU_3090], separateCapable: true })}
          placementMode="pooled"
        />,
      );

      expect(screen.getByTestId('vram-container-node-a')).toBeInTheDocument();
      expect(screen.queryByTestId('vram-container-node-a::gpu-0')).not.toBeInTheDocument();
      expect(screen.queryByTestId('vram-container-node-a::gpu-1')).not.toBeInTheDocument();
    });

    it('does not render GPU-indexed dropzones', () => {
      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, GPU_3090], separateCapable: true })}
          placementMode="pooled"
        />,
      );

      expect(screen.queryByTestId('node-node-a-gpu-0-dropzone')).not.toBeInTheDocument();
      expect(screen.queryByTestId('node-node-a-gpu-1-dropzone')).not.toBeInTheDocument();
    });
  });

  describe('separate mode', () => {
    it('renders one dropzone per GPU target', () => {
      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, GPU_3090], separateCapable: true })}
          placementMode="separate"
        />,
      );

      expect(screen.getByTestId('node-node-a-gpu-0-dropzone')).toBeInTheDocument();
      expect(screen.getByTestId('node-node-a-gpu-1-dropzone')).toBeInTheDocument();
      expect(screen.queryByTestId('node-node-a-pool-dropzone')).not.toBeInTheDocument();
    });

    it('labels each GPU container with the gpu target label via hostname', () => {
      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, GPU_3090], separateCapable: true })}
          placementMode="separate"
        />,
      );

      const gpu0 = within(screen.getByTestId('node-node-a-gpu-0-dropzone'))
        .getByTestId('vram-container-node-a::gpu-0');
      const gpu1 = within(screen.getByTestId('node-node-a-gpu-1-dropzone'))
        .getByTestId('vram-container-node-a::gpu-1');

      expect(gpu0).toHaveAttribute('data-hostname', 'GPU 0 · RTX 4090 · 24.0 GB');
      expect(gpu1).toHaveAttribute('data-hostname', 'GPU 1 · RTX 3090 · 24.0 GB');
    });

    it('passes per-GPU VRAM capacity to each container', () => {
      const gpuSmall: GpuTarget = {
        index: 1,
        name: 'RTX 3060',
        vramBytes: 12_000_000_000,
        reservedBytes: 0,
        label: 'GPU 1 · RTX 3060 · 12.0 GB',
      };

      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, gpuSmall], separateCapable: true })}
          placementMode="separate"
          totalVramBytes={36_000_000_000}
        />,
      );

      const gpu0 = within(screen.getByTestId('node-node-a-gpu-0-dropzone'))
        .getByTestId('vram-container-node-a::gpu-0');
      const gpu1 = within(screen.getByTestId('node-node-a-gpu-1-dropzone'))
        .getByTestId('vram-container-node-a::gpu-1');

      expect(gpu0).toHaveAttribute('data-total-vram', '24000000000');
      expect(gpu1).toHaveAttribute('data-total-vram', '12000000000');
    });

    it('filters assignments by GPU index to the correct container', () => {
      const assignments = [
        makeVramAssignment({ id: 'Qwen3::gpu-0', name: 'Qwen3' }),
        makeVramAssignment({ id: 'GLM::gpu-1', name: 'GLM' }),
      ];

      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, GPU_3090], separateCapable: true })}
          placementMode="separate"
          assignments={assignments}
        />,
      );

      const gpu0zone = screen.getByTestId('node-node-a-gpu-0-dropzone');
      const gpu1zone = screen.getByTestId('node-node-a-gpu-1-dropzone');

      expect(within(gpu0zone).getByTestId('mock-assignment-Qwen3::gpu-0')).toBeInTheDocument();
      expect(within(gpu0zone).queryByTestId('mock-assignment-GLM::gpu-1')).not.toBeInTheDocument();

      expect(within(gpu1zone).getByTestId('mock-assignment-GLM::gpu-1')).toBeInTheDocument();
      expect(within(gpu1zone).queryByTestId('mock-assignment-Qwen3::gpu-0')).not.toBeInTheDocument();
    });

    it('routes split assignments by GPU index', () => {
      const assignments = [
        makeVramAssignment({ id: 'Qwen3::abc123::0-23-48::gpu-0', name: 'Qwen3' }),
        makeVramAssignment({ id: 'Qwen3::abc123::24-47-48::gpu-1', name: 'Qwen3' }),
      ];

      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({ gpuTargets: [GPU_4090, GPU_3090], separateCapable: true })}
          placementMode="separate"
          assignments={assignments}
        />,
      );

      const gpu0zone = screen.getByTestId('node-node-a-gpu-0-dropzone');
      const gpu1zone = screen.getByTestId('node-node-a-gpu-1-dropzone');

      expect(within(gpu0zone).getByTestId('mock-assignment-Qwen3::abc123::0-23-48::gpu-0')).toBeInTheDocument();
      expect(within(gpu1zone).getByTestId('mock-assignment-Qwen3::abc123::24-47-48::gpu-1')).toBeInTheDocument();
    });
  });

  describe('mixed GPU warning', () => {
    it('shows warning banner when mixedGpuWarning is true in separate mode', () => {
      const crossGpuAssignments = [
        makeVramAssignment({ id: 'Qwen3::abc123::0-23-48::gpu-0', sizeBytes: 11_000_000_000 }),
        makeVramAssignment({ id: 'Qwen3::abc123::24-47-48::gpu-1', sizeBytes: 11_000_000_000 }),
      ];
      render(
        <ConfigNodeSection
          {...baseProps}
          assignments={crossGpuAssignments}
          node={makeNode({
            gpuTargets: [GPU_4090, GPU_3090],
            separateCapable: true,
            mixedGpuWarning: true,
          })}
          placementMode="separate"
        />,
      );

      expect(screen.getByTestId('node-node-a-warning-mixed-gpu')).toBeInTheDocument();
    });

    it('shows warning banner even in pooled mode', () => {
      render(
        <ConfigNodeSection
          {...baseProps}
          node={makeNode({
            gpuTargets: [GPU_4090, GPU_3090],
            separateCapable: true,
            mixedGpuWarning: true,
          })}
          placementMode="pooled"
        />,
      );

      expect(screen.getByTestId('node-node-a-warning-mixed-gpu')).toBeInTheDocument();
    });

    it('does not show warning when mixedGpuWarning is false', () => {
      render(
        <ConfigNodeSection {...baseProps} node={makeNode({ mixedGpuWarning: false })} placementMode="pooled" />,
      );

      expect(screen.queryByTestId('node-node-a-warning-mixed-gpu')).not.toBeInTheDocument();
    });

     it('warning text describes the mixed GPU condition', () => {
       const crossGpuAssignments = [
         makeVramAssignment({ id: 'Qwen3::abc123::0-23-48::gpu-0', sizeBytes: 11_000_000_000 }),
         makeVramAssignment({ id: 'Qwen3::abc123::24-47-48::gpu-1', sizeBytes: 11_000_000_000 }),
       ];
       render(
         <ConfigNodeSection
           {...baseProps}
           assignments={crossGpuAssignments}
           node={makeNode({
             gpuTargets: [GPU_4090, GPU_3090],
             separateCapable: true,
             mixedGpuWarning: true,
           })}
           placementMode="separate"
         />,
       );

       const warning = screen.getByTestId('node-node-a-warning-mixed-gpu');
       expect(warning).toHaveTextContent(/mixed/i);
     });

     it('warning text mentions PCIe bandwidth limitation', () => {
       render(
         <ConfigNodeSection
           {...baseProps}
           node={makeNode({
             gpuTargets: [GPU_4090, GPU_3090],
             separateCapable: true,
             mixedGpuWarning: true,
           })}
           assignments={[
             makeVramAssignment({ id: 'Qwen3::node-a::gpu-0', name: 'Qwen3' }),
             makeVramAssignment({ id: 'Qwen3::node-a::gpu-1', name: 'Qwen3' }),
           ]}
           placementMode="separate"
         />,
       );

       const warning = screen.getByTestId('node-node-a-warning-mixed-gpu');
       expect(warning).toHaveTextContent(/PCIe/i);
       expect(warning).toHaveTextContent(/P2P/i);
     });
   });

  describe('defaults', () => {
    it('defaults to pooled mode when placementMode is omitted', () => {
      render(
        <ConfigNodeSection {...baseProps} node={makeNode()} />,
      );

      expect(screen.getByTestId('node-node-a-pool-dropzone')).toBeInTheDocument();
    });
  });
});
