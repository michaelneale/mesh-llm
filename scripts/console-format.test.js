const test = require("node:test");
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");

const { extractDetails, format } = require("./console-format.js");

test("extractDetails keeps invite token details", () => {
  const details = extractDetails({
    event: "invite_token",
    level: "info",
    token: "tok_123",
    mesh_id: "mesh_abc",
    message: "invite token ready",
  });

  assert.deepEqual(details, ["token=tok_123", "mesh=mesh_abc"]);
});

test("extractDetails renders native debug params", () => {
  const details = extractDetails({
    event: "model",
    level: "debug",
    message: "Reading model metadata...",
    architecture: "qwen35",
    ctx: 262144,
    blocks: 32,
  });

  assert.deepEqual(details, [
    "architecture=qwen35",
    "ctx=262144",
    "blocks=32",
  ]);
});

test("extractDetails renders tensor group params for native debug events", () => {
  const details = extractDetails({
    event: "model",
    level: "debug",
    message: "Reading tensor groups...",
    f32: 177,
    q4_K: 74,
    q5_K: 69,
  });

  assert.deepEqual(details, ["f32=177", "q4_K=74", "q5_K=69"]);
});

test("extractDetails ignores native params on non-debug events", () => {
  const details = extractDetails({
    event: "model",
    level: "info",
    message: "Reading model metadata...",
    architecture: "qwen35",
  });

  assert.deepEqual(details, []);
});

test("format appends native debug params on followup line", () => {
  const rendered = format({
    timestamp: "2026-05-11T03:48:48.392Z",
    level: "debug",
    event: "model",
    message: "Reading model metadata...",
    architecture: "qwen35",
    ctx: 262144,
  });

  assert.match(rendered, /DEBUG.*Reading model metadata/);
  assert.match(rendered, /↳ architecture=qwen35 \| ctx=262144/);
});

test("extractDetails captures shutdown_requested signal", () => {
  const details = extractDetails({
    event: "SIGINT",
    level: "info",
    signal: "SIGINT",
    message: "shutdown requested (SIGINT)",
  });

  assert.deepEqual(details, ["signal=SIGINT"]);
});

test("extractDetails captures shutdown reason", () => {
  const details = extractDetails({
    event: "shutdown",
    level: "info",
    reason: "user requested shutdown",
    message: "mesh-llm shutting down",
  });

  assert.deepEqual(details, ["reason=user requested shutdown"]);
});

test("extractDetails captures model unloading details", () => {
  const details = extractDetails({
    event: "model_unloading",
    level: "info",
    model: "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
    message: "unloading model unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
  });

  assert.deepEqual(details, ["model=unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL"]);
});

test("formatter drains shutdown records after SIGINT", async () => {
  const child = spawn(process.execPath, ["scripts/console-format.js"], {
    cwd: __dirname + "/..",
    stdio: ["pipe", "pipe", "pipe"],
  });

  let output = "";
  let errorOutput = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    output += chunk;
  });
  child.stderr.on("data", (chunk) => {
    errorOutput += chunk;
  });

  child.stdin.write(JSON.stringify({
    timestamp: "2026-05-12T03:18:32.334Z",
    level: "info",
    event: "ready",
    message: "mesh-llm runtime ready",
  }) + "\n");

  await new Promise((resolve) => child.stdout.once("data", resolve));

  child.kill("SIGINT");

  await new Promise((resolve) => setTimeout(resolve, 25));

  child.stdin.write(JSON.stringify({
    timestamp: "2026-05-12T03:18:47.666Z",
    level: "info",
    event: "SIGINT",
    signal: "SIGINT",
    message: "shutdown requested (SIGINT)",
  }) + "\n");
  child.stdin.write(JSON.stringify({
    timestamp: "2026-05-12T03:18:47.666Z",
    level: "info",
    event: "shutdown",
    reason: "user requested shutdown",
    message: "mesh-llm shutting down",
  }) + "\n");
  child.stdin.end();

  const exit = await new Promise((resolve) => {
    child.on("close", (code, signal) => resolve({ code, signal }));
  });

  assert.deepEqual(exit, { code: 0, signal: null });
  assert.equal(errorOutput, "");
  assert.match(output, /INFO.*mesh-llm runtime ready/);
  assert.match(output, /INFO.*shutdown requested \(SIGINT\)/);
  assert.match(output, /↳ signal=SIGINT/);
  assert.match(output, /INFO.*mesh-llm shutting down/);
  assert.match(output, /↳ reason=user requested shutdown/);
});
