# Private Meshes

A private mesh lets your own machines discover each other without joining the public mesh.

Create or join a named mesh:

```sh
mesh-llm serve --discover "team-lab"
```

Join from another serving machine:

```sh
mesh-llm serve --discover "team-lab"
```

Join as an API-only client:

```sh
mesh-llm client --discover "team-lab"
```

Use private meshes for lab machines, office workstations, or a home cluster where the models and traffic should stay within your own mesh.
