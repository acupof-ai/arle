// Docs sidebar navigation. Each group is a section header followed by a
// list of links. Hrefs are absolute paths under BASE_URL (consumer
// prepends `import.meta.env.BASE_URL`); items pointing at unbuilt pages
// are kept in the nav for shape but href to GitHub or "#".

export type DocsLink = {
  label: string;
  href: string;        // relative to BASE_URL when starting with "docs/"
  badge?: "new" | "beta";
  active?: boolean;
};

export type DocsGroup = {
  heading: string;
  links: DocsLink[];
};

export const DOCS_NAV: DocsGroup[] = [
  {
    heading: "Get started",
    links: [
      { label: "Introduction", href: "https://github.com/cklxx/arle/blob/main/README.md" },
      { label: "Install", href: "#install" },
      { label: "Quickstart", href: "docs/", active: true },
      { label: "First serve", href: "#serve" },
      { label: "Verify with --doctor", href: "#verify" },
    ],
  },
  {
    heading: "Concepts",
    links: [
      { label: "Front door & verbs", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
      { label: "Runtime spine", href: "https://github.com/cklxx/arle/tree/main/infer/src" },
      { label: "Backend dispatch", href: "https://github.com/cklxx/arle/blob/main/infer/src/backend/AGENTS.md" },
      { label: "Stability levels", href: "https://github.com/cklxx/arle/blob/main/docs/stability-policy.md" },
    ],
  },
  {
    heading: "Run",
    links: [
      { label: "arle run", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
      { label: "arle serve", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
      { label: "arle train", href: "https://github.com/cklxx/arle/tree/main/crates/train", badge: "beta" },
      { label: "arle data", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
    ],
  },
  {
    heading: "Backends",
    links: [
      { label: "CUDA", href: "https://github.com/cklxx/arle/tree/main/crates/cuda-kernels" },
      { label: "Metal", href: "https://github.com/cklxx/arle/tree/main/crates/mlx-sys", badge: "beta" },
      { label: "CPU", href: "https://github.com/cklxx/arle/blob/main/docs/support-matrix.md" },
      { label: "Kernel crates", href: "https://github.com/cklxx/arle/tree/main/crates", badge: "new" },
    ],
  },
  {
    heading: "Operate",
    links: [
      { label: "Environment vars", href: "https://github.com/cklxx/arle/blob/main/docs/environment.md" },
      { label: "Metrics", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
      { label: "Sessions", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
      { label: "Logs & tracing", href: "https://github.com/cklxx/arle/blob/main/docs/environment.md" },
    ],
  },
];

export const DOCS_TABS = [
  { label: "Guides", href: "docs/", active: true },
  { label: "HTTP API", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
  { label: "CLI reference", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
  { label: "Bench snapshots", href: "https://github.com/cklxx/arle/tree/main/docs/experience/wins" },
  { label: "Recipes", href: "https://github.com/cklxx/arle/tree/main/examples" },
  { label: "Operator", href: "https://github.com/cklxx/arle/blob/main/docs/environment.md" },
];
