import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


out_dir = Path(os.environ["OUT_DIR"])
model_path = os.environ["MODEL"]
corpus_path = Path(os.environ["CORPUS"])
llama = os.environ["LLAMA"]
port = int(os.environ.get("PORT", "18080"))
max_tokens = int(os.environ.get("MAX_TOKENS", "192"))
prompt_limit = int(os.environ.get("PROMPT_LIMIT", "8"))
ctx_size = os.environ.get("CTX_SIZE", "4096")
warmup_requests = int(os.environ.get("WARMUP_REQUESTS", "2"))
warmup_tokens = int(os.environ.get("WARMUP_TOKENS", "64"))
corpus_warmup_tokens = int(os.environ.get("CORPUS_WARMUP_TOKENS", "0"))
windows = [
    int(value)
    for value in os.environ.get("WINDOWS", "1,2,3,4,5").split(",")
    if value.strip()
]
include_baseline = os.environ.get("INCLUDE_BASELINE", "1") not in {"0", "false", "False"}

out_dir.mkdir(parents=True, exist_ok=True)

prompts = []
with corpus_path.open() as f:
    for line in f:
        if len(prompts) >= prompt_limit:
            break
        if not line.strip():
            continue
        prompts.append(json.loads(line))

conditions = []
if include_baseline:
    conditions.append({"name": "baseline_warm", "args": ["--spec-type", "none"]})
for window in windows:
    conditions.append(
        {
            "name": f"mtp_w{window}_warm",
            "args": [
                "--spec-type",
                "mtp",
                "--spec-draft-n-max",
                str(window),
                "--spec-draft-n-min",
                "1",
            ],
        }
    )


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode()) if raw else None


def wait_ready():
    last = None
    for _ in range(900):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status < 500:
                    return
        except Exception as e:
            last = e
        time.sleep(1)
    raise RuntimeError(f"server not ready: {last}")


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def run_completion(model_id, prompt, tokens):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": False,
    }
    return http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload, timeout=900)


for cond in conditions:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = out_dir / f"llama-server-{cond['name']}-{ts}.log"
    result_path = out_dir / f"llama-server-{cond['name']}-{ts}.json"
    cmd = [
        llama,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        ctx_size,
        "-np",
        "1",
        "-ngl",
        "999",
        "--no-webui",
    ] + cond["args"]
    print(f"=== starting {cond['name']} ===", flush=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready()
        model_id = "llama-mtp-pr"
        try:
            models = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=10)
            if models and models.get("data"):
                model_id = models["data"][0].get("id") or model_id
        except Exception:
            pass

        for index in range(warmup_requests):
            print(f"warmup {index + 1}/{warmup_requests}", flush=True)
            run_completion(
                model_id,
                "Reply with exactly one short sentence confirming the model is ready.",
                warmup_tokens,
            )
        if corpus_warmup_tokens > 0 and prompts:
            print("corpus warmup 1/1", flush=True)
            run_completion(model_id, prompts[0]["prompt"], corpus_warmup_tokens)

        results = []
        for rec in prompts:
            start = time.perf_counter()
            error = None
            completion_tokens = 0
            prompt_tokens = 0
            content_len = 0
            try:
                value = run_completion(model_id, rec["prompt"], max_tokens)
                usage = value.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens") or 0)
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                content = ((value.get("choices") or [{}])[0].get("message") or {}).get(
                    "content"
                ) or ""
                content_len = len(content)
            except Exception as e:
                error = repr(e)
            elapsed = time.perf_counter() - start
            row = {
                "prompt_id": rec.get("prompt_id"),
                "category": rec.get("category"),
                "latency_ms": elapsed * 1000,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_tok_s": completion_tokens / elapsed
                if elapsed and completion_tokens
                else 0,
                "content_chars": content_len,
                "error": error,
            }
            print(json.dumps(row), flush=True)
            results.append(row)

        ok = [r for r in results if not r["error"]]
        lat = [r["latency_ms"] for r in ok]
        comp = sum(r["completion_tokens"] for r in ok)
        elapsed_sum = sum(r["latency_ms"] for r in ok) / 1000
        report = {
            "condition": cond["name"],
            "server": "llama.cpp PR 22673 llama-server",
            "model_path": model_path,
            "prompt_corpus": str(corpus_path),
            "prompt_count": len(prompts),
            "max_tokens": max_tokens,
            "warmup_requests": warmup_requests,
            "warmup_tokens": warmup_tokens,
            "corpus_warmup_tokens": corpus_warmup_tokens,
            "concurrency_depth": 1,
            "sampling": {"temperature": 0, "enable_thinking": None},
            "spec_args": cond["args"],
            "summary": {
                "ok": len(ok),
                "errors": len(results) - len(ok),
                "mean_latency_ms": statistics.mean(lat) if lat else None,
                "p50_latency_ms": percentile(lat, 0.50),
                "p95_latency_ms": percentile(lat, 0.95),
                "completion_tokens": comp,
                "completion_tok_s": comp / elapsed_sum if elapsed_sum else 0,
            },
            "results": results,
            "server_log": str(log_path),
        }
        result_path.write_text(json.dumps(report, indent=2))
        print(f"wrote {result_path}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        time.sleep(3)
