import { test, expect } from '@playwright/test';

const mockStatus = {
  version: '0.51.0',
  node_id: 'test-node-1',
  token: 'test-token-abc',
  node_status: 'idle',
  is_host: false,
  is_client: false,
  llama_ready: false,
  model_name: '',
  serving_models: [],
  api_port: 9337,
  my_vram_gb: 8.0,
  model_size_gb: 0,
  mesh_name: null,
  peers: [],
  mesh_models: [],
  inflight_requests: 0,
  nostr_discovery: false,
  my_hostname: 'test-host',
  my_is_soc: true,
  gpus: [{ name: 'Apple M1', vram_bytes: 8589934592 }],
};

test.describe('Smoke Tests @smoke', () => {
  test.beforeEach(async ({ page }) => {
    // Mock /api/status
    await page.route('**/api/status', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockStatus),
      }),
    );

    // Mock /api/events SSE — abort since we don't need live updates
    await page.route('**/api/events', (route) => route.abort());

    // Mock /api/config — used by ConfigPage on load
    await page.route('**/api/config', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '',
      }),
    );
  });

  test('dashboard renders with mesh visualization', async ({ page }) => {
    await page.goto('/dashboard');

    // Navigation should be visible
    await expect(page.getByRole('link', { name: 'Network' })).toBeVisible();

    // Dashboard alert shows private mesh status
    await expect(page.getByText('Your private mesh')).toBeVisible();
  });

  test('displays version info from mocked status', async ({ page }) => {
    await page.goto('/dashboard');

    // Footer renders the version returned by the mocked /api/status
    await expect(page.getByText(/Mesh LLM v0\.51\.0/)).toBeVisible();
  });

  test('can navigate to chat section', async ({ page }) => {
    await page.goto('/dashboard');

    await page.getByRole('link', { name: 'Chat' }).click();

    await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible();
  });

  test('can navigate to config section', async ({ page }) => {
    await page.goto('/dashboard');

    // Click the Configuration nav link
    await page.getByRole('link', { name: 'Configuration' }).click();

    // Config page renders its heading
    await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible();
  });

  test('navigation links are all present', async ({ page }) => {
    await page.goto('/dashboard');

    await expect(page.getByRole('link', { name: 'Network' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Chat' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Configuration' })).toBeVisible();
  });

  test('chat section renders directly via URL', async ({ page }) => {
    await page.goto('/chat');

    await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible();
  });

  test('config section renders directly via URL', async ({ page }) => {
    await page.goto('/config');

    await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible();
  });

  test('v3 separate-mode config rehydrates with separate placement active', async ({ page }) => {
    await page.route('**/api/status', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...mockStatus,
          owner_id: 'owner-1',
          gpus: [
            { name: 'RTX 4090', vram_bytes: 24_000_000_000 },
            { name: 'RTX 3080', vram_bytes: 10_000_000_000 },
          ],
        }),
      }),
    );

    const separateToml = [
      'version = 1',
      '',
      '[[nodes]]',
      'node_id = "test-node-1"',
      'placement_mode = "separate"',
      '',
      '[[nodes.models]]',
      'name = "Qwen3"',
      'gpu_index = 1',
    ].join('\n');

    await page.route('**/api/config', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: separateToml,
      }),
    );

    await page.goto('/config');

    await expect(page.getByRole('heading', { name: 'Configuration' })).toBeVisible();
    await expect(page.getByTestId('node-test-node-1-mode-separate')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('node-test-node-1-gpu-1-dropzone')).toBeVisible();
  });

});
