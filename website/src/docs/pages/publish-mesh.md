# Publish Mesh

A publish mesh exposes a reachable endpoint for a mesh you operate. Use it when you want other clients or machines to join a known mesh without relying on local discovery.

Start a serving node:

```sh
mesh-llm serve --auto
```

Share the published endpoint or invite details with the machines that should join:

```sh
mesh-llm client --auto
```

For production deployments, keep the public endpoint as an API entry point and keep model-serving machines behind the mesh boundary.
