#!/usr/bin/env python3
import argparse
import json
import time
from urllib import request


PROMPTS = [
    {
        "name": "code_python",
        "prompt": "Write a Python function that returns the n-th Fibonacci number using memoization. Include a docstring.",
    },
    {
        "name": "code_cpp",
        "prompt": "Write a C++ template function `clamp(x, lo, hi)` that returns x clamped to [lo, hi]. No std::clamp.",
    },
    {
        "name": "explain_concept",
        "prompt": "Explain how speculative decoding works in large language model inference, in three short paragraphs.",
    },
    {
        "name": "summarize",
        "prompt": "Summarize in two sentences: The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing through mechanization, steam power, and the factory system. It spread to continental Europe and North America during the 19th century.",
    },
    {"name": "qa_factual", "prompt": "Q: What are the four fundamental forces of physics?\nA:"},
    {"name": "translation", "prompt": "Translate to French: 'The quick brown fox jumps over the lazy dog.'"},
    {"name": "creative_short", "prompt": "Write a four-line poem about an old lighthouse."},
    {
        "name": "stepwise_math",
        "prompt": "Solve step by step: A train leaves station A at 60 km/h. Two hours later, a second train leaves the same station on the same track at 90 km/h. How long until the second train catches the first?",
    },
    {
        "name": "long_code_review",
        "prompt": (
            "You are reviewing a backend service that has been suffering intermittent latency spikes "
            "in production. Below is the relevant code and a description of the system. After reading "
            "carefully, produce a structured review with three sections: (1) likely root causes ranked "
            "by probability, (2) concrete code or configuration changes you would make first, "
            "(3) what telemetry you would add to confirm the diagnosis.\n\n"
            "System description: a Python FastAPI service in front of a Postgres 15 database, deployed "
            "as four replicas behind an nginx load balancer. Each request reads a user record, fetches "
            "their last 50 events from a partitioned events table, computes an aggregate score, writes "
            "the score back to the user row, and returns a JSON response. Average payload is 4 KB. "
            "p50 latency is 35 ms; p99 spikes to 1.8 seconds approximately every 90 seconds in a "
            "regular pattern. The spikes correlate with elevated Postgres CPU but not with elevated "
            "Postgres connection count. The application pool is sized at 20 connections per replica. "
            "PgBouncer is in front of Postgres in transaction pooling mode with a pool size of 50.\n\n"
            "Code excerpt - the hot endpoint:\n"
            "```python\n@app.post('/score/{user_id}')\nasync def score(user_id: int, payload: ScoreRequest):\n"
            " async with db.transaction() as tx:\n user = await tx.fetchrow(\n"
            " 'SELECT id, tier, last_score FROM users WHERE id = $1 FOR UPDATE',\n user_id,\n )\n"
            " if user is None:\n raise HTTPException(404)\n events = await tx.fetch(\n"
            " 'SELECT type, weight, ts FROM events '\n 'WHERE user_id = $1 ORDER BY ts DESC LIMIT 50',\n user_id,\n )\n"
            " new_score = compute_score(user['tier'], events, payload.signals)\n"
            " await tx.execute(\n 'UPDATE users SET last_score = $1, updated_at = now() WHERE id = $2',\n new_score, user_id,\n )\n"
            " await tx.execute(\n 'INSERT INTO score_history (user_id, score, ts) VALUES ($1, $2, now())',\n user_id, new_score,\n )\n"
            " await cache.set(f'score:{user_id}', new_score, ex=300)\n"
            " metrics.histogram('score.latency_ms').observe((time.time() - start) * 1000)\n"
            " return {'user_id': user_id, 'score': new_score}\n```\n\n"
            "Schema notes: `users` is ~50M rows, `events` is partitioned by month with ~2B rows total "
            "and a btree index on `(user_id, ts DESC)`. `score_history` is unpartitioned, ~800M rows, "
            "with a single index on `user_id`. Postgres autovacuum is at default settings. There is "
            "a nightly batch job that rebuilds materialized views starting at 02:00 UTC; spikes occur "
            "throughout the day, not just during the batch window. Connection pooling metrics show "
            "PgBouncer waiting connections occasionally hit 8-12 during spikes but never saturate. "
            "CPU on the FastAPI replicas stays below 30% even during spikes. Network round-trip time "
            "between the application and Postgres is consistently 0.4 ms.\n\nBegin your review now."
        ),
    },
]


def post(url, payload):
    req = request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())


def run(args):
    out = {"results": []}
    for prompt in PROMPTS:
        start = time.time()
        response = post(
            f"{args.url}/completion",
            {
                "prompt": prompt["prompt"],
                "n_predict": args.n_predict,
                "temperature": 0.0,
                "seed": 42,
                "cache_prompt": False,
                "stream": False,
            },
        )
        wall = time.time() - start
        timings = response.get("timings", {}) or {}
        row = {
            "name": prompt["name"],
            "wall_s": round(wall, 3),
            "predicted_n": timings.get("predicted_n"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n", 0),
            "draft_n_accepted": timings.get("draft_n_accepted", 0),
        }
        row["accept_rate"] = (
            round(row["draft_n_accepted"] / row["draft_n"], 4) if row["draft_n"] else None
        )
        out["results"].append(row)
        accept_rate = f"{row['accept_rate']:.3f}" if row["accept_rate"] is not None else "n/a"
        print(
            f" {row['name']:<18} pred={row['predicted_n']:>4} "
            f"draft={row['draft_n']:>4} acc={row['draft_n_accepted']:>4} "
            f"rate={accept_rate} tok/s={row['predicted_per_second']:.1f}",
            flush=True,
        )

    total_draft = sum(row["draft_n"] or 0 for row in out["results"])
    total_accepted = sum(row["draft_n_accepted"] or 0 for row in out["results"])
    total_predicted = sum(row["predicted_n"] or 0 for row in out["results"])
    total_wall = sum(row["wall_s"] for row in out["results"])
    out["aggregate"] = {
        "n_requests": len(out["results"]),
        "total_predicted": total_predicted,
        "total_draft": total_draft,
        "total_draft_accepted": total_accepted,
        "aggregate_accept_rate": round(total_accepted / total_draft, 4) if total_draft else None,
        "wall_s_total": round(total_wall, 2),
    }
    print("\nAggregate:", json.dumps(out["aggregate"], indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("Wrote", args.out, flush=True)


def diff(a, b):
    left, right = json.load(open(a)), json.load(open(b))
    print(f"{'metric':<24} {'A':>14} {'B':>14} {'delta':>10}")
    for key in (
        "aggregate_accept_rate",
        "total_predicted",
        "total_draft",
        "total_draft_accepted",
        "wall_s_total",
    ):
        va, vb = left["aggregate"].get(key), right["aggregate"].get(key)
        if va is None or vb is None:
            print(f"{key:<24} {str(va):>14} {str(vb):>14}")
            continue
        delta = vb - va
        rendered = f"{delta:>+10.4f}" if isinstance(delta, float) else f"{delta:>+10}"
        print(f"{key:<24} {va:>14} {vb:>14} {rendered}")


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8080")
parser.add_argument("--out")
parser.add_argument("--diff", nargs=2)
parser.add_argument("--n-predict", type=int, default=192)
args = parser.parse_args()
if args.diff:
    diff(*args.diff)
else:
    run(args)
