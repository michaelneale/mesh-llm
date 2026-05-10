# Mesh LLM Website

The public website is built with Eleventy and emitted into `../docs` for static hosting.

```sh
cd website
npm install
npm run build
```

During development:

```sh
cd website
npm run dev
```

From the repository root:

```sh
just website-dev
just website-build
```

`npm run dev` runs Eleventy with watch mode, incremental builds, and browser
reload on port 8765.

Source files live in `website/src`:

- `index.njk` - landing page
- `catalog/index.njk` - live Hugging Face catalog page
- `docs/index.njk` - docs landing page
- `docs/pages/*.md` - public documentation pages
- `_includes/` - shared layouts, nav, footer, and hero visual
- `assets/site.css` - shared styling

The generated static output lives in `docs/`.
