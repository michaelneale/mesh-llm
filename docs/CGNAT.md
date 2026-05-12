# CGNAT and Direct Mesh Connectivity

mesh-llm direct peer connections use UDP. Relays can keep a mesh usable, but
direct UDP is the path you want for lower latency and predictable split
inference. If direct connections work in one direction but not the other, check
for carrier-grade NAT (CGNAT) before chasing application-level issues.

## Symptom

A typical CGNAT failure looks like this:

- The local router has a port-forward rule, such as `UDP 53238 -> 192.168.1.50:53238`.
- The local machine advertises a public-looking internet address in its invite token.
- A remote peer cannot reach the forwarded UDP port.
- Raw UDP tests from the remote peer time out.
- Relay paths may still work, or direct paths may work only from the non-CGNAT side.

This can be asymmetric. One network may have a real public IPv4 address on its
router, while the other router only has a CGNAT address.

## What CGNAT Is

With normal home NAT, your router owns the public IPv4 address:

```text
internet
  -> public IPv4 on router
  -> LAN device, for example 192.168.1.50
```

With CGNAT, the ISP adds another NAT layer in front of your router:

```text
internet
  -> ISP shared public IPv4
  -> router WAN address from 100.64.0.0/10
  -> LAN device, for example 192.168.1.50
```

Your router can only forward ports from its own WAN address to a LAN device. It
cannot create a port forward through the ISP's CGNAT layer. That means a router
port-forward rule may be correct and still be unreachable from the internet.

## Diagnostic Pattern

Compare the IPv4 address the internet sees with the IPv4 address your router
believes is its WAN address.

On the mesh host:

```bash
curl -4 https://ifconfig.me
```

Then check your router's WAN address in its admin UI or via UPnP/IGD if your
router exposes it. A CGNAT case looks like this:

```text
internet-visible IPv4: 198.51.100.10
router WAN IPv4:       100.72.12.34
```

Any router WAN IPv4 in `100.64.0.0/10` is CGNAT space. Other private ranges,
such as `10.0.0.0/8`, `172.16.0.0/12`, and `192.168.0.0/16`, also mean the
router does not directly own the public IPv4 address.

A healthy direct-forwarding setup looks like this:

```text
internet-visible IPv4: 198.51.100.10
router WAN IPv4:       198.51.100.10
```

## Raw UDP Probe

Before debugging mesh-llm itself, prove that inbound UDP reaches the host. Run a
listener on the local mesh host:

```bash
python3 - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 53238))
print("listening on UDP 53238", flush=True)
while True:
    data, peer = sock.recvfrom(2048)
    print(f"received {len(data)} bytes from {peer}", flush=True)
    sock.sendto(b"ack:" + data, peer)
PY
```

From a remote network, send packets to the public IPv4 and forwarded UDP port:

```bash
python3 - <<'PY'
import socket
import time

target = ("198.51.100.10", 53238)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)

for i in range(6):
    payload = f"udp-probe-{i}".encode()
    sock.sendto(payload, target)
    try:
        data, peer = sock.recvfrom(2048)
    except socket.timeout:
        print(f"timeout waiting for reply {i}")
    else:
        print(f"reply from {peer}: {data.decode(errors='replace')}")
    time.sleep(1)
PY
```

If the listener receives nothing, the failure is below mesh-llm. Check router
port-forwarding, CGNAT, local firewall policy, and upstream ISP filtering before
debugging QUIC or mesh protocol behavior.

## mesh-llm Check

When testing a fixed UDP port, bind mesh-llm to that port so the advertised
candidate and router forward agree:

```bash
mesh-llm serve \
  --model hf://example/model-layers \
  --split \
  --bind-port 53238
```

The peer status should report a direct path:

```text
peer_count=1
latency_source=direct
```

If UDP probing works but mesh-llm still falls back to relay, inspect the invite
token candidates and mesh logs. If the token advertises private or CGNAT
addresses ahead of a usable public candidate, direct dialing may fail or take
longer than expected.

## Fixes

For CGNAT:

- Ask the ISP to disable CGNAT, opt out of CGNAT, or provide a static public
  IPv4 address.
- Reboot or reconnect the router after the ISP change so it renews its WAN
  lease.
- Re-check that the router WAN IPv4 matches the internet-visible IPv4.
- Keep the router UDP port-forward rule pointed at the mesh host.

For dual-router setups:

- Put the upstream device into bridge mode, or
- Add matching UDP forwards at every NAT layer, from the public edge to the mesh
  host.

For overlapping private LANs:

- Do not rely on private `192.168.x.x` candidates across sites.
- Prefer a real public IPv4 candidate, IPv6 when both networks have working
  global IPv6, or an overlay network that mesh-llm can actually dial.

