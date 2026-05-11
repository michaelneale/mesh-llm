# Updating Mesh

Run the installer again to move to the latest released build:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | sh
```

Restart the local node after updating:

```sh
mesh-llm stop
mesh-llm serve --auto
```

If the machine is only an API client:

```sh
mesh-llm stop
mesh-llm client --auto
```

Mixed-version meshes should continue to operate during rolling updates. Update serving nodes one at a time when the mesh is actively handling traffic.
