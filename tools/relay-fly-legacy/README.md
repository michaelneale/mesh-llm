# ⚠️ LEGACY — Not in use

This was a self-hosted iroh-relay on Fly.io. It is **no longer used**.

mesh-llm now uses managed iroh relay infrastructure via [services.iroh.computer](https://services.iroh.computer):

| Relay | Region | URL |
|-------|--------|-----|
| USW1-2 | US West | `https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./` |
| APS1-1 | Asia-Pacific South | `https://aps1-1.relay.michaelneale.mesh-llm.iroh.link./` |

These are configured as defaults in `mesh-llm/src/mesh/mod.rs`.

The old Fly.io relay (`mesh-llm-relay.fly.dev`) can be decommissioned.

---

<details>
<summary>Original README (archived)</summary>

# iroh-relay on Fly.io

Self-hosted [iroh-relay](https://github.com/n0-computer/iroh/tree/main/iroh-relay) for mesh-llm. Provides relay connectivity for peers that can't connect directly (symmetric NAT, firewalls, etc).

**Live**: `https://mesh-llm-relay.fly.dev/`

## Architecture

```
  iroh client                    Fly.io edge                  container
  ─────────                      ──────────                   ─────────
  https://mesh-llm-relay.fly.dev ──► TLS termination ──► plain HTTP :8080
       WebSocket upgrade         ◄── passes through  ◄── iroh-relay (no TLS)
       binary relay framing
```

## Deploy

```bash
cd relay/
fly deploy
```

## Configuration

See `iroh-relay.toml`. Key settings:

| Setting | Value | Notes |
|---------|-------|-------|
| `http_bind_addr` | `[::]:8080` | Fly forwards here after TLS termination |
| `enable_relay` | `true` | The actual relay service |
| `enable_quic_addr_discovery` | `false` | Can't work behind Fly's proxy, not needed |
| `access` | `everyone` | Open relay |
| Rate limit | 2 MB/s per client | Generous for relay traffic |

## Infra

- Region: `syd`
- 2 machines (Fly default HA), auto-stop when idle
- shared-cpu-1x, 512MB RAM

</details>
