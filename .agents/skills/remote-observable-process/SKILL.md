---
name: remote-observable-process
description: Use this skill when starting, supervising, debugging, holding open, or stopping any remote process over SSH that needs an operator-like interactive environment, a TTY, login-shell startup files, long-running observation, logs, readiness checks, or later inspection.
metadata:
  short-description: Run observable SSH processes
---

# remote-observable-process

Use this skill before starting a non-trivial remote process over SSH when the
process needs to behave like it was launched by an operator in a terminal, be
observed after launch, expose readiness, keep running beyond one command, or be
stopped cleanly later.

## Rule

For environment-sensitive processes, prefer SSH with a TTY and an interactive
login shell:

```bash
ssh -tt host '/bin/zsh -ilc '\''COMMAND'\'''
```

If the remote host does not use zsh, adapt the shell while preserving the same
properties: allocate a TTY, use a login/interactive shell, and keep the session
foreground for first repro/debug runs.

Avoid detached first attempts such as:

```bash
ssh host "nohup COMMAND > /tmp/process.log 2>&1 &"
ssh host "COMMAND > /tmp/process.log 2>&1 &"
```

Those shapes hide lifecycle and can behave differently from a real remote
session.

## Stage Runtime Debugging

For network-sensitive server chains, model stage servers, GPU/Metal workloads,
or bind/connect debugging, first prove the process in a held foreground TTY:

```bash
ssh -tt host '/bin/zsh -ilc '\''COMMAND 2>&1 | tee /tmp/COMMAND.log'\'''
```

Keep the SSH session open while sending traffic. Use `tmux` or `screen` only
after validating that they preserve the same listener, downstream connection,
and request behavior on that host.
