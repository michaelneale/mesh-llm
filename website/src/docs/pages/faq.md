# FAQ

## Is Mesh a model provider?

Mesh is a way to serve local and mesh-ready models across machines through one OpenAI-compatible endpoint.

## Do I need multiple machines?

No. Mesh can serve from one machine. Multiple machines matter when you want to pool capacity or run models that benefit from layer packages.

## What is a layer package?

A layer package is the public artifact Mesh uses to place parts of a model across machines for multi-machine serving.

## Can I use existing agent tools?

Yes. Point OpenAI-compatible tools at `http://localhost:3131/v1`.

## How do models appear in the Catalog?

Catalog entries are contributed through metadata and validation evidence. Layer packages can be local first or hosted on Hugging Face, then referenced by a catalog pull request.

