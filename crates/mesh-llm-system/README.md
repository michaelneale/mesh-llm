# mesh-llm-system

`mesh-llm-system` owns machine-local concerns for mesh-llm.

This crate includes:

- backend flavor and binary/device helpers
- hardware discovery and GPU identity/facts
- process liveness and PID validation helpers
- local benchmark fingerprinting and prompt corpus import support
- release target and self-update plumbing

Keep distributed mesh membership, request routing, API routes, CLI dispatch, and
host runtime orchestration outside this crate. Those layers may consume system
facts, but this crate should stay focused on local platform behavior.
