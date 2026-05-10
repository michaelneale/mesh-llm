# Troubleshooting

## Check Mesh status

```sh
curl http://localhost:3131/api/status
```

## Verify the local API endpoint

```sh
curl http://localhost:3131/v1/models
```

## Stop stale processes

```sh
mesh-llm stop
```

If a development instance is wedged, use the project cleanup commands from the testing docs.

## Public mesh connection issues

If this machine cannot join the public mesh, verify network access, relay connectivity, and that only one local Mesh instance is running.

