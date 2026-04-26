import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://cklxx.github.io",
  base: "/arle/",
  outDir: "./dist",
  trailingSlash: "ignore",
  build: { format: "directory", inlineStylesheets: "always" },
  compressHTML: true,
});
