# Mesh website plan

This plan defines the first public website for `meshllm.cloud`. The site should
feel like the public face of the Mesh console: useful, technical, compact, and
operational. It should not read like a generic AI SaaS landing page.

## Goals

- Explain Mesh quickly: multi-machine model serving through one OpenAI-compatible endpoint.
- Make installation and first use obvious.
- Promote the public mesh at `public.meshllm.cloud`.
- Make the Catalog a top-level product surface for models, Hugging Face links,
  and layer packages.
- Turn the existing project docs into a user-facing documentation system.
- Use the same visual language as the console: shadcn-style primitives, dense
  cards, badges, command blocks, tables, and restrained borders.

## Non-goals

- Do not expose internal engine names in primary website copy.
- Do not use ideological public-utility positioning.
- Do not lead with architecture diagrams before users understand what they can do.
- Do not make a decorative marketing splash page.
- Do not create a separate brand language from the Mesh console.

## Public terminology

| Concept | Public term | Notes |
|---|---|---|
| Serving one model across machines | Multi-machine model serving | Primary capability name |
| Model pieces used for multi-machine serving | Layer packages | User-facing artifact name |
| Models that can run well on Mesh | Mesh-ready models | Catalog badge |
| Shared running Mesh network | Public Mesh | Use for `public.meshllm.cloud` |
| Model discovery surface | Catalog | Top-level nav item |
| OpenAI API compatibility | OpenAI-compatible endpoint | Important for agents |

Avoid using internal engine names in top-level website copy. Advanced internal
docs can still mention implementation names where they are necessary for
contributors.

## Top navigation

```text
[Mesh]  Catalog  Docs  Public Mesh  GitHub              Install
```

Desktop nav should keep `Install` as the primary button. `Public Mesh` should be
visible because it is a differentiator, but it should not compete with install
as the main call to action.

Mobile nav should collapse into a sheet-style menu using the same behavior as
the console.

## Homepage outline

1. Hero
   - Headline: "Run large models across every machine you own"
   - Supporting copy: "Mesh pools machines into a distributed inference network,
     so you can serve large models across multiple devices through one
     OpenAI-compatible endpoint."
   - Primary CTA: Install Mesh
   - Secondary CTA: Open Public Mesh
   - Install command block

2. Status strip
   - Public Mesh
   - OpenAI-compatible API
   - macOS / Linux / Windows
   - Multi-machine model serving
   - Layer packages

3. How Mesh works
   - Machines join a mesh.
   - Mesh serves models through one endpoint.
   - Large models can use layer packages across multiple machines.

4. Catalog preview
   - Top-level link to `/catalog/`.
   - Show sample rows/cards for model name, capabilities, runtime, source,
     Hugging Face link, and whether layer packages are available.

5. Public Mesh
   - Explain `public.meshllm.cloud` as a running Mesh network.
   - CTA to open it and docs link for joining it.

6. Agents and tools
   - Goose
   - Pi
   - opencode
   - OpenAI-compatible clients

7. Documentation entry points
   - Get started
   - Running models
   - Meshes
   - Catalog
   - Integrations
   - Reference

## Documentation information architecture

Inspired by the Ollama docs shape, Mesh docs should optimize for the user path:
quickstart first, install by platform, model usage, capabilities, API/reference,
integrations, and help. Mesh-specific concepts such as public/private meshes,
layer packages, and Catalog contribution get first-class sections.

```text
Docs
├── Get started
│   ├── Quickstart
│   ├── Installing Mesh
│   └── Update Mesh
│
├── Install
│   ├── macOS
│   ├── Linux
│   ├── Windows
│   └── Hardware support
│
├── Running models
│   ├── Run your first model
│   ├── Multi-machine model serving
│   ├── Layer packages
│   └── Catalog
│
├── Capabilities
│   ├── OpenAI-compatible API
│   ├── Streaming
│   ├── Tool calling
│   └── Structured outputs
│
├── Meshes
│   ├── Join the public mesh
│   ├── Create a private mesh
│   └── Publish your own mesh
│
├── Catalog
│   ├── Browse Catalog
│   ├── Contributing layer packages
│   └── Certifying model families
│
├── Integrations
│   ├── Integrations overview
│   ├── Agent setup
│   ├── exo comparison
│   └── Plugins
│
├── API reference
│   ├── API reference
│   ├── OpenAI-compatible API
│   └── CLI reference
│
├── Help
│   ├── FAQ
│   ├── Troubleshooting
│   └── Testing playbook
│
└── Contributing
    ├── Contributing guide
    ├── Testing playbook
    └── Roadmap
```

## Catalog requirements

The Catalog should become a product surface, not just a documentation page.
It should answer:

- What models can I run with Mesh?
- Which models are Mesh-ready?
- Which models have layer packages?
- Which models are visible on the public mesh?
- Where are the Hugging Face model and layer-package repos?
- What hardware or mesh size do I need?
- What command do I run?
- How do I contribute layer packages back to the Catalog?
- How do I certify a new model architecture or family?

### Catalog list fields

- Model name
- Family/provider
- Size and quantization
- Capabilities: text, vision, tools, reasoning, MoE
- Runtime: single-machine or multi-machine
- Badges: Mesh-ready, layer packages available, public mesh available
- Source: Mesh catalog, Hugging Face, community layer repo
- Recommended hardware or number of machines
- Run command

### Catalog detail page tabs

```text
Overview | Run | Layer packages | Hardware | API | Hugging Face
```

### Hugging Face integration

Initial static version:

- Link out to Hugging Face model repos.
- Link out to Hugging Face layer-package repos.
- Keep metadata in a checked-in JSON or generated static file.
- Read the live `meshllm/catalog` dataset through the Hugging Face dataset repo
  API and generated Dataset Viewer rows.

Later dynamic version:

- Fetch catalog data from the Mesh catalog source.
- Hydrate Hugging Face metadata at build time.
- Generate `/catalog/:model` pages.
- Emit `/llms.txt` and machine-readable model metadata for agents.

### Hugging Face Dataset Viewer

The runtime catalog keeps nested `entries/**/*.json` files because that shape is
good for model resolution. Hugging Face Dataset Viewer needs a flat table, so the
catalog publish flow should also emit:

- `catalog_rows.jsonl`: one row per model variant;
- `README.md` dataset-card YAML with `configs.default.data_files` pointing at
  `catalog_rows.jsonl`.
- An hourly GitHub Actions workflow should regenerate these files from the live
  `entries/**/*.json` tree so Hugging Face dataset PR merges cannot leave the
  Dataset Viewer stale.

That makes these APIs usable for the website:

```text
https://huggingface.co/api/datasets/meshllm/catalog
https://datasets-server.huggingface.co/rows?dataset=meshllm%2Fcatalog&config=default&split=train
https://datasets-server.huggingface.co/parquet?dataset=meshllm%2Fcatalog
```

## Layer package contribution docs

The docs need to support two contribution paths:

1. Local artifacts
   - Build or materialize layer packages locally.
   - Validate manifest structure, checksums, tensor ranges, layer ranges, and
     model compatibility.
   - Run a local smoke test against Mesh before opening a contribution.
   - Generate the Catalog metadata patch.

2. Hugging Face hosted artifacts
   - Publish layer packages to a Hugging Face repo.
   - Include expected repo layout, manifest files, model source links, license
     notes, and validation metadata.
   - Reference the Hugging Face repo from the Catalog entry.
   - Open a PR that updates Catalog metadata without committing large artifacts
     to the Mesh repo.

The PR workflow should be explicit:

- Contributor creates or identifies layer packages.
- Contributor runs validation.
- Contributor updates Catalog metadata.
- CI verifies metadata, Hugging Face links, manifests, checksums, and basic
  compatibility.
- Maintainers review model naming, license/source clarity, hardware guidance,
  and validation evidence.
- After merge, the model appears in the public Catalog as Mesh-ready or layer
  package available.

## Model family certification docs

Certification docs should explain how a new architecture or model family becomes
trusted for Mesh:

- Identify architecture, tokenizer behavior, GGUF metadata, layer naming, and
  tensor layout.
- Confirm layer package generation understands the family.
- Validate single-machine execution first.
- Validate multi-machine model serving across representative split boundaries.
- Check output parity or acceptable tolerance against full-model execution.
- Record supported quantizations, known limits, and hardware guidance.
- Promote the family into Catalog policy only after evidence is attached.

The docs should distinguish:

- A model entry: one model or quantization listed in the Catalog.
- A layer package: artifacts enabling multi-machine serving for a model.
- A certified family: an architecture or family Mesh knows how to split,
  validate, and support predictably.

## Visual system

Use shadcn-style building blocks and the existing console visual language:

- Buttons for primary commands.
- Cards for repeated docs/catalog entries.
- Badges for model capabilities and runtime state.
- Tabs for OS install commands and model detail pages.
- Tables for catalog, hardware, and compatibility data.
- Command blocks for install and agent setup.
- Sheet-style mobile navigation.
- Accordion for FAQ and troubleshooting.

Keep the style restrained:

- Neutral dark background.
- Thin borders.
- Compact spacing.
- 8px radius or less.
- Minimal animation.
- No decorative gradients, orbs, or generic AI imagery.

## Static site direction

The website and documentation should be statically rendered together. Raw
Markdown links are not acceptable for the public site because they do not render
as a coherent documentation experience on the hosted domain.

Target shape:

- One source tree for the landing page, Catalog, and docs.
- Markdown/MDX docs rendered into HTML at build time.
- Shared layout, nav, footer, search, and syntax highlighting.
- Catalog pages generated from metadata.
- Output published as the `mesh-llm.cloud` static site.

The current implementation uses Eleventy from `website/src` and writes generated
static output into `docs/`.

```sh
cd website
npm install
npm run build
```

Eleventy is intentionally lightweight for the first website: it renders Markdown
docs, shared Nunjucks layouts, shared CSS, and the live Catalog page without
adding a client runtime.

## Initial scaffold

The current scaffold keeps GitHub Pages compatibility by using static files under
`docs/`:

- `docs/index.html` - landing page
- `docs/catalog/index.html` - first Catalog surface
- `docs/CNAME` - existing custom domain

The files under `docs/` are generated output plus static hosting assets. Edit
`website/src` for website changes, then rebuild.

## Possible future stack

- Keep Eleventy while the site is mostly static Markdown and lightweight
  JavaScript.
- Consider Astro or another component SSG if the site needs richer component
  islands.
- Consider MDX if docs need embedded interactive components.
- Keep shadcn-style primitives and lucide-style icons when the site gains a
  component framework.
- Build-time catalog generation from checked-in metadata and Hugging Face.

## First implementation milestones

1. Replace the existing landing page with the new navigation, copy, and Catalog
   entry points.
2. Add the static Catalog scaffold.
3. Replace raw Markdown public links with rendered docs pages.
4. Add first-pass docs pages for install, public mesh, private meshes,
   multi-machine model serving, layer packages, contributing layer packages,
   model family certification, and agents.
5. Publish and maintain the first-party `https://mesh-llm.cloud/install.sh`
   installer route used by the homepage quickstart.
6. Add `/llms.txt` so agents can discover the public docs.
7. Add catalog metadata and generate real Catalog rows from it.
8. Move live Catalog data from client-side fetch to build-time generation if SEO
   or availability requires static rows.
