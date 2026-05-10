# Contributing Layer Packages

Layer packages let Mesh place a model across multiple machines without every node downloading the full model. A package records the source model, quantization, layer artifacts, and validation metadata.

## Local contribution flow

Create or validate a package locally, then publish the package repository to Hugging Face.

```sh
mesh-llm models show unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
```

After publishing, open a pull request against the catalog dataset entry. The PR should include:

- Source model repo and revision.
- Source GGUF filename and quantization.
- Layer package repo.
- Package manifest metadata.
- Validation result.

## Hugging Face contribution flow

If the package is generated on Hugging Face infrastructure, publish the package repository first. Then submit the catalog change as a dataset PR to:

[meshllm/catalog](https://huggingface.co/datasets/meshllm/catalog)

When the PR is merged, the hourly catalog refresh regenerates `catalog_rows.jsonl`. The website catalog then picks up the new row through the Hugging Face Dataset Viewer API.

## Catalog PR behavior

Catalog PRs should be reviewable as metadata changes. The website reads flattened catalog rows, but the source of truth remains the catalog dataset entries and package repositories.
