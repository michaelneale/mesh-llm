export default {
  layout: "doc.njk",
  eleventyComputed: {
    title: (data) =>
      data.title ||
      data.page.fileSlug
        .replace(/-/g, " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase()),
  },
  permalink: "/docs/pages/{{ page.fileSlug }}/index.html",
};
