#!/usr/bin/env node

const readline = require("readline");

// ANSI colors
const c = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  green: "\x1b[32m",
  blue: "\x1b[34m",
  cyan: "\x1b[36m",
};

const NATIVE_DEBUG_EVENTS = new Set(["backend", "model", "memory", "kv_cache", "tokenizer"]);
const STRUCTURAL_KEYS = new Set(["timestamp", "level", "event", "message"]);

function levelColor(level) {
  switch ((level || "").toLowerCase()) {
    case "error": return c.red;
    case "warn": return c.yellow;
    case "info": return c.green;
    case "debug": return c.cyan;
    default: return c.dim;
  }
}

function fmtTime(ts) {
  if (!ts) return "";
  try {
    return new Date(ts).toISOString().replace("T", " ").replace("Z", "");
  } catch {
    return ts;
  }
}

function formatValue(value) {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function extractDetails(obj) {
  const details = [];

  if (obj.event === "invite_token" && obj.token) {
    details.push(`token=${obj.token}`);
    if (obj.mesh_id) details.push(`mesh=${obj.mesh_id}`);
  }

  if (obj.port) details.push(`port=${obj.port}`);
  if (obj.http_port) details.push(`http_port=${obj.http_port}`);
  if (obj.internal_port) details.push(`internal_port=${obj.internal_port}`);
  if (obj.model) details.push(`model=${obj.model}`);
  if (obj.api_url) details.push(`api=${obj.api_url}`);
  if (obj.signal) details.push(`signal=${obj.signal}`);
  if (obj.reason) details.push(`reason=${obj.reason}`);
  if (obj.context) details.push(obj.context);

  if (String(obj.level || "").toLowerCase() === "debug" && NATIVE_DEBUG_EVENTS.has(obj.event)) {
    for (const [key, value] of Object.entries(obj)) {
      if (STRUCTURAL_KEYS.has(key)) continue;
      if (["port", "http_port", "internal_port", "model", "api_url", "context", "token"].includes(key)) {
        continue;
      }
      details.push(`${key}=${formatValue(value)}`);
    }
  }

  return details;
}

function format(obj) {
  const ts = fmtTime(obj.timestamp);
  const level = (obj.level || "info").toUpperCase();
  const msg = obj.message || obj.event || "—";

  const color = levelColor(level);

  let out = `${c.dim}${ts}${c.reset} - ${color}${level}${c.reset}: ${msg}`;

  // ---- DETAIL EXTRACTION ----
  const details = extractDetails(obj);

  if (details.length) {
    out += `\n  ${c.blue}↳ ${details.join(" | ")}${c.reset}`;
  }

  return out;
}

function main() {
  // In a shell pipeline, Ctrl-C sends SIGINT to every foreground process.
  // mesh-llm handles the signal and emits final shutdown JSONL; this formatter
  // must stay alive long enough to drain stdin until mesh-llm closes the pipe.
  const ignoreSignalUntilInputCloses = () => {};
  process.on("SIGINT", ignoreSignalUntilInputCloses);
  process.on("SIGTERM", ignoreSignalUntilInputCloses);

  const rl = readline.createInterface({
    input: process.stdin,
    terminal: false,
  });

  rl.on("line", (line) => {
    if (!line.trim()) return;

    try {
      const obj = JSON.parse(line);
      console.log(format(obj));
    } catch {
      console.log(line);
    }
  });

  rl.on("close", () => {
    process.off("SIGINT", ignoreSignalUntilInputCloses);
    process.off("SIGTERM", ignoreSignalUntilInputCloses);
  });
}

if (require.main === module) {
  main();
}

module.exports = { extractDetails, format, formatValue, main };
