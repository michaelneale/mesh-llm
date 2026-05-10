# Running Large Models Across Machines

Mesh can serve models that are too large for one device by placing different parts of the model on different machines. Requests still go through one local OpenAI-compatible endpoint.

Start every machine that can contribute compute:

```sh
mesh-llm serve --auto
```

Then serve a model from the catalog:

```sh
mesh-llm serve --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
```

When a model has layer packages, Mesh can place the required layers across available machines instead of requiring the full model on every node.

Use the catalog to find models with layer packages:

```sh
mesh-llm models search gemma
```

The website catalog shows package availability and copyable model refs in `org/repo:quant` form.
