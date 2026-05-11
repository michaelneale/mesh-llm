import markdownItAnchor from "markdown-it-anchor";

export default function(eleventyConfig) {
  eleventyConfig.addPassthroughCopy("src/mesh-llm-logo.svg");
  eleventyConfig.addPassthroughCopy("src/mesh.png");
  eleventyConfig.addPassthroughCopy("src/CNAME");
  eleventyConfig.addPassthroughCopy("src/assets");

  eleventyConfig.amendLibrary("md", (md) => {
    md.use(markdownItAnchor, {
      permalink: false,
      slugify: (value) =>
        String(value)
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/(^-|-$)/g, ""),
    });
  });

  eleventyConfig.addFilter("json", (value) => JSON.stringify(value));
  eleventyConfig.addTransform("trim-trailing-whitespace", (content) =>
    typeof content === "string" ? content.replace(/[ \t]+$/gm, "") : content
  );

  return {
    dir: {
      input: "src",
      includes: "_includes",
      output: "../docs",
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    templateFormats: ["md", "njk", "html"],
  };
}
