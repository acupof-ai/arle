// Content for both ARLE landing pages. Each page imports its locale block
// and feeds the same components — visual identity is shared, copy is not.

export type Signal = {
  kv: string; // raw HTML allowed: e.g. `backend=<code>cuda</code>`
  status: "ok" | "warn";
};

export type Surface = { cap: string; body: string };
export type StatusRow = { cap: string; body: string };

export type MatrixCell = string; // raw HTML allowed
export type MatrixRow = MatrixCell[];
export type Topology = {
  title: string;
  legend: { cap: string; body: string; href?: string }[];
};
export type Matrix = {
  title: string;
  caption: string;
  head: string[];
  rows: MatrixRow[];
  note: string;
};

export type Install = {
  title: string;
  caption: string;
  cards: {
    label: string; // small kicker shown above the command
    lines: string[]; // raw HTML allowed
  }[];
  note: string;
};

export type BenchGridCell = {
  label: string;
  value: string; // numeric, no unit
  unit: string;  // short unit like "tok/s", "ms"
  /** v3 — sparkline SVG path on a 64×18 viewBox. Optional. */
  spark?: string;
};
export type BenchRow = {
  date: string;
  backend: string;
  model: string;
  hardware: string;
  metric: string; // raw HTML allowed (legacy short blob)
  href: string;
  /** v2 split-card body — 4-cell metric grid */
  grid?: BenchGridCell[];
  /** v2 card footer — run command */
  cmd?: string;
  /** v2 card header pill */
  stability?: "stable" | "beta" | "dev";
};
export type Bench = {
  title: string;
  caption: string;
  head: string[];
  rows: BenchRow[];
  note: string;
};

export type QuickCard = {
  title: string;
  lines: string[];
};

export type FileRow = {
  href: string;
  path: string;
  desc: string;
};

export type FooterCol = {
  heading: string;
  links: { label: string; href: string; placeholder?: boolean }[];
};

/** v3 hero — large display heading composed of HTML lines, plus a lede paragraph. */
export type HeroDisplay = {
  /** Each entry is an HTML line of the H1 (allows <span class="strike">, <em>). */
  h1Lines: string[];
  /** Paragraph below the H1 (raw HTML allowed). */
  lede: string;
  /** Status label rendered in the kicker bar (e.g. "v0.4.2 stable"). */
  status: string;
  /** v3 — short uppercase annotation under the H1 (e.g. "Pure Rust · OpenAI v1"). */
  replPrimary?: string;
  /** v3 — serif italic suffix on the repl-line (e.g. "— CUDA + Metal + CPU"). */
  replSuffix?: string;
};

/** v3 hero-right perf cells. */
export type HeroStat = {
  lbl: string;
  num: string;
  unit: string;
  meta: string; // raw HTML allowed
  /** v3 — sparkline SVG path on a 60×14 viewBox. Optional. */
  spark?: string;
};

/** v2 sticky top-nav link. */
export type TopNavLink = { label: string; href: string };

export type Locale = {
  lang: string;
  hreflang: string;
  meta: {
    title: string;
    description: string;
    ogTitle: string;
    ogDescription: string;
    ogUrl: string;
    canonical: string;
  };
  termtabs: {
    on: string;
    cwd: string;
    meta: string;
  };
  manhead: { left: string; center: string; right: string };
  manfoot: { left: string; center: string; right: string };
  hero: {
    kicker: string;
    tagline: string; // raw HTML allowed
    signals: Signal[];
    primaryCta: string;
    secondaryCta: string;
    /** v2 split layout — large display heading + lede + perf grid. */
    display?: HeroDisplay;
    /** v2 top-nav primary links (left of github). */
    nav?: TopNavLink[];
    /** v2 hero-right install block: render the first install card by default;
     *  pass a label override here. */
    installLabel?: string;
    /** v2 hero-right perf grid — 4 cells. */
    stats?: HeroStat[];
    /** v2 build version chip (e.g. "v0.4.2"). */
    version?: string;
  };
  jumps: { label: string; href: string }[];
  langSwitch: {
    other: { label: string; href: string; hreflang: string };
    selfLabel: string; // bold marker for current locale
  };
  sections: {
    name: { title: string; body: string; note: string };
    install: Install;
    glance: {
      title: string;
      cards: { h: string; body: string }[];
    };
    synopsis: { title: string; lines: string[] };
    topology: Topology;
    surfaces: { title: string; rows: Surface[] };
    matrix: Matrix;
    status: { title: string; rows: StatusRow[] };
    bench: Bench & {
      /** v2 frame heading (used in place of the small caps title). */
      v2Heading?: string;
    };
    quickstart: {
      title: string;
      cards: QuickCard[];
      note: string;
    };
    examples: {
      title: string;
      lead: string;
      lines: string[];
      seeAlso: string;
    };
    files: { title: string; rows: FileRow[] };
    seealso: {
      title: string;
      items: string[];
      note: string;
    };
  };
  docfoot: {
    cols: FooterCol[];
    meta: string;
  };
};

const FILES_EN = [
  { href: "https://github.com/cklxx/arle/blob/main/README.md", path: "/README.md", desc: "public overview · install · CLI · architecture" },
  { href: "https://github.com/cklxx/arle/blob/main/README.zh-CN.md", path: "/README.zh-CN.md", desc: "simplified Chinese public entry" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md", path: "/docs/http-api.md", desc: "HTTP contract · streaming behavior" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/support-matrix.md", path: "/docs/support-matrix.md", desc: "backend / model / quant support" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/stability-policy.md", path: "/docs/stability-policy.md", desc: "stability levels · compatibility posture" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/compatibility.md", path: "/docs/compatibility.md", desc: "compatibility and deprecation policy" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/environment.md", path: "/docs/environment.md", desc: "environment variables · operator defaults" },
  { href: "https://github.com/cklxx/arle/tree/main/examples", path: "/examples", desc: "copyable curl · Docker · Metal · tiny train smokes" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/index.md", path: "/docs/index.md", desc: "maintainer-facing PARA index" },
  { href: "https://github.com/cklxx/arle/releases", path: "/releases", desc: "tagged binaries · checksums" },
];

const FILES_ZH = [
  { href: "https://github.com/cklxx/arle/blob/main/README.zh-CN.md", path: "/README.zh-CN.md", desc: "中文公共入口：安装 · CLI · 架构" },
  { href: "https://github.com/cklxx/arle/blob/main/README.md", path: "/README.md", desc: "英文公共入口" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md", path: "/docs/http-api.md", desc: "HTTP 契约 · 流式行为" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/support-matrix.md", path: "/docs/support-matrix.md", desc: "后端 / 模型 / 量化支持矩阵" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/stability-policy.md", path: "/docs/stability-policy.md", desc: "稳定性分级 · 兼容性姿态" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/compatibility.md", path: "/docs/compatibility.md", desc: "兼容性与废弃策略" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/environment.md", path: "/docs/environment.md", desc: "环境变量与默认值" },
  { href: "https://github.com/cklxx/arle/tree/main/examples", path: "/examples", desc: "curl · Docker · Metal · tiny train 冒烟示例" },
  { href: "https://github.com/cklxx/arle/blob/main/docs/index.md", path: "/docs/index.md", desc: "维护者 PARA 索引" },
  { href: "https://github.com/cklxx/arle/releases", path: "/releases", desc: "发版二进制 · 校验和" },
];

const SIGNALS: Signal[] = [
  { kv: 'backend=<code>cuda</code>', status: "ok" },
  { kv: 'backend=<code>metal</code>', status: "warn" },
  { kv: 'api=<code>openai-v1</code>', status: "ok" },
  { kv: 'doors=<code>1</code>', status: "ok" },
];

const DOCFOOT_EN: FooterCol[] = [
  {
    heading: "Documentation",
    links: [
      { label: "Getting Started", href: "https://github.com/cklxx/arle/blob/main/README.md" },
      { label: "HTTP API", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
      { label: "Examples", href: "https://github.com/cklxx/arle/tree/main/examples" },
      { label: "Quickstart", href: "#quickstart" },
    ],
  },
  {
    heading: "Project",
    links: [
      { label: "About", href: "https://github.com/cklxx/arle" },
      { label: "Architecture", href: "https://github.com/cklxx/arle/blob/main/docs/architecture.md" },
      { label: "License", href: "https://github.com/cklxx/arle/blob/main/LICENSE" },
      { label: "Code of Conduct", href: "https://github.com/cklxx/arle/blob/main/CODE_OF_CONDUCT.md" },
    ],
  },
  {
    heading: "Community",
    links: [
      { label: "GitHub Issues", href: "https://github.com/cklxx/arle/issues" },
      { label: "Discussions", href: "https://github.com/cklxx/arle/discussions" },
      { label: "Security", href: "https://github.com/cklxx/arle/blob/main/SECURITY.md" },
    ],
  },
  {
    heading: "Status",
    links: [
      { label: "Roadmap", href: "https://github.com/cklxx/arle/blob/main/ROADMAP.md" },
      { label: "Changelog", href: "https://github.com/cklxx/arle/blob/main/CHANGELOG.md" },
      { label: "Releases", href: "https://github.com/cklxx/arle/releases" },
      { label: "Bench Snapshots", href: "https://github.com/cklxx/arle/tree/main/docs/experience/wins" },
    ],
  },
];

const DOCFOOT_ZH: FooterCol[] = [
  {
    heading: "文档",
    links: [
      { label: "快速开始", href: "https://github.com/cklxx/arle/blob/main/README.zh-CN.md" },
      { label: "HTTP API", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
      { label: "示例", href: "https://github.com/cklxx/arle/tree/main/examples" },
      { label: "Quickstart", href: "#quickstart" },
    ],
  },
  {
    heading: "项目",
    links: [
      { label: "项目主页", href: "https://github.com/cklxx/arle" },
      { label: "架构", href: "https://github.com/cklxx/arle/blob/main/docs/architecture.md" },
      { label: "许可证", href: "https://github.com/cklxx/arle/blob/main/LICENSE" },
      { label: "Code of Conduct", href: "https://github.com/cklxx/arle/blob/main/CODE_OF_CONDUCT.md" },
    ],
  },
  {
    heading: "社区",
    links: [
      { label: "GitHub Issues", href: "https://github.com/cklxx/arle/issues" },
      { label: "Discussions", href: "https://github.com/cklxx/arle/discussions" },
      { label: "安全策略", href: "https://github.com/cklxx/arle/blob/main/SECURITY.md" },
    ],
  },
  {
    heading: "状态",
    links: [
      { label: "Roadmap", href: "https://github.com/cklxx/arle/blob/main/ROADMAP.md" },
      { label: "Changelog", href: "https://github.com/cklxx/arle/blob/main/CHANGELOG.md" },
      { label: "发版", href: "https://github.com/cklxx/arle/releases" },
      { label: "Bench 快照", href: "https://github.com/cklxx/arle/tree/main/docs/experience/wins" },
    ],
  },
];

export const EN: Locale = {
  lang: "en",
  hreflang: "en",
  meta: {
    title: "ARLE(1) — runtime-first Rust workspace",
    description:
      "ARLE is the runtime-first Rust workspace for serving, local agents, training, evaluation, and dataset tooling.",
    ogTitle: "ARLE — runtime-first Rust workspace",
    ogDescription:
      "One Rust workspace for infer serving, the arle front door, train/eval flows, dataset tooling, and public docs.",
    ogUrl: "https://cklxx.github.io/arle/",
    canonical: "https://cklxx.github.io/arle/",
  },
  termtabs: { on: "man arle.1", cwd: "~/code/arle", meta: "zsh · 104×42" },
  manhead: { left: "ARLE(1)", center: "General Commands Manual", right: "ARLE(1)" },
  manfoot: { left: "ARLE(1)", center: "April 2026", right: "ARLE(1)" },
  hero: {
    kicker: "runtime-first rust workspace",
    tagline:
      '<span class="hl">infer</span> serves OpenAI-compatible traffic. <span class="hl">arle</span> operates the local agent, train, eval, and data flows. One Rust workspace.',
    signals: SIGNALS,
    primaryCta: "Quickstart",
    secondaryCta: "cklxx/arle",
    version: "v0.4.2",
    nav: [
      { label: "Docs", href: "docs/" },
      { label: "Bench", href: "#bench" },
      { label: "Install", href: "#install" },
      { label: "Examples", href: "#examples" },
    ],
    display: {
      status: "v0.4.2 stable",
      h1Lines: [
        'Serve <span class="strike"><span class="txt">Qwen3</span></span>',
        "at the speed",
        "of <em>cargo build</em>.",
      ],
      replPrimary: "Pure Rust · OpenAI v1",
      replSuffix: "— CUDA + Metal + CPU",
      lede:
        '<b>infer</b> serves OpenAI-compatible traffic on CUDA, Metal, and CPU. <b>arle</b> is the unified front door for run, serve, train, and data flows. <a class="inline" href="#topology">One Rust workspace &mdash; zero glue.</a>',
    },
    installLabel: "install · macos · linux",
    stats: [
      {
        lbl: "cuda · L4 · qwen3-4b",
        num: "118",
        unit: "tok/s",
        meta: 'conc=<b>16</b> · BF16',
        spark: "M0 9 L8 8 L16 10 L24 7 L32 6 L40 7 L48 4 L60 3",
      },
      {
        lbl: "cuda · L4 · ITL p50",
        num: "59.9",
        unit: "ms",
        meta: '<b>2026-04-23</b>',
        spark: "M0 5 L8 6 L16 5 L24 8 L32 7 L40 9 L48 10 L60 11",
      },
      {
        lbl: "metal · M4 Pro · 0.8b q4_k_m",
        num: "211",
        unit: "tok/s",
        meta: 'gguf decode · <b>2026-04-27</b>',
        spark: "M0 11 L8 9 L16 7 L24 6 L32 5 L40 4 L48 3 L60 2",
      },
      {
        lbl: "cold start · arle serve",
        num: "1.4",
        unit: "s",
        meta: "load + first token",
        spark: "M0 5 L8 6 L16 5 L24 5 L32 4 L40 5 L48 4 L60 4",
      },
    ],
  },
  jumps: [
    { label: "[install]", href: "#install" },
    { label: "[glance]", href: "#glance" },
    { label: "[topology]", href: "#topology" },
    { label: "[matrix]", href: "#matrix" },
    { label: "[bench]", href: "#bench" },
    { label: "[quickstart]", href: "#quickstart" },
    { label: "[synopsis]", href: "#synopsis" },
    { label: "[examples]", href: "#examples" },
    { label: "[docs]", href: "#docs" },
    { label: "[github]", href: "https://github.com/cklxx/arle" },
    { label: "[releases]", href: "https://github.com/cklxx/arle/releases" },
  ],
  langSwitch: {
    other: { label: "zh", href: "zh-cn/", hreflang: "zh-Hans" },
    selfLabel: "en",
  },
  sections: {
    name: {
      title: "NAME",
      body:
        "<b>arle</b> — runtime-first Rust workspace. <code>infer</code> serves; <code>arle</code> operates.",
      note:
        "This page is the stable entry. Product truth lives in the repo docs linked below.",
    },
    install: {
      title: "INSTALL",
      caption:
        "One runnable line per platform. Full matrix &amp; env-var overrides in <a href=\"https://github.com/cklxx/arle/blob/main/docs/install.md\">docs/install.md</a>.",
      cards: [
        {
          label: "Apple Silicon · Homebrew",
          lines: [
            '<span class="p">$</span> brew install cklxx/tap/arle',
            '<span class="p">$</span> arle --doctor',
          ],
        },
        {
          label: "Apple Silicon / Linux x86_64 · curl",
          lines: [
            '<span class="p">$</span> curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh \\',
            '    | sh',
            '<span class="p">$</span> arle --doctor',
          ],
        },
        {
          label: "CUDA · GPU container",
          lines: [
            '<span class="p">$</span> docker run --rm --gpus all -p 8000:8000 \\',
            '    -v /path/to/Qwen3-4B:/model:ro \\',
            '    ghcr.io/cklxx/arle:latest \\',
            '    serve --backend cuda --model-path /model',
          ],
        },
        {
          label: "Source · Linux / Apple Silicon",
          lines: [
            '<span class="p">$</span> git clone https://github.com/cklxx/arle &amp;&amp; cd arle',
            '<span class="p">$</span> ./setup.sh',
            '<span class="p">$</span> cargo build --release --features cli --bin arle',
            '<span class="p">$</span> ./target/release/arle serve --model-path /path/to/Qwen3-4B --port 8000',
          ],
        },
      ],
      note:
        'Pre-built tarballs &amp; SHAs live on each <a href="https://github.com/cklxx/arle/releases">GitHub Release</a>; the curl installer verifies SHA256 before extracting. Apple Silicon source build: swap <code>--features cli</code> → <code>--features metal,no-cuda,cli</code>.',
    },
    glance: {
      title: "AT A GLANCE",
      cards: [
        {
          h: "— Runtime Spine",
          body:
            "Serving remains primary. <code>infer</code> is the canonical HTTP surface for generation, metrics, stats, and sessions.",
        },
        {
          h: "— One Front Door",
          body:
            "<code>arle run</code>, <code>arle serve</code>, <code>arle train</code>, and <code>arle data</code> give users one stable front door instead of a scatter of task-specific binaries.",
        },
        {
          h: "— Shared Authority",
          body:
            "Runtime code, model loading, train/eval flows, and public docs stay in one Rust workspace so operator truth does not drift.",
        },
      ],
    },
    synopsis: {
      title: "SYNOPSIS",
      lines: [
        '<span class="ln">1</span><span class="p">$</span> cargo build -p infer --release',
        '<span class="ln">2</span><span class="p">$</span> ./target/release/infer --model-path /path/to/Qwen3-4B --port 8000',
        '<span class="ln">3</span><span class="p">$</span> cargo build --release --features cli --bin arle',
        '<span class="ln">4</span><span class="p">$</span> ./target/release/arle --doctor',
        '<span class="ln">5</span><span class="p">$</span> ./target/release/arle --model-path /path/to/Qwen3-4B \\',
        '<span class="ln"> </span>    run --prompt <i>"Summarize this repo"</i>',
        '<span class="ln">6</span><span class="p">$</span> ./target/release/arle serve --backend cuda \\',
        '<span class="ln"> </span>    --model-path /path/to/Qwen3-4B --port 8000',
        '<span class="ln">7</span><span class="p">$</span> ./target/release/arle train env',
        '<span class="ln">8</span><span class="p">$</span> ./target/release/arle data convert --help<span class="caret"></span>',
      ],
    },
    topology: {
      title: "TOPOLOGY",
      legend: [
        { cap: "front door", href: "https://github.com/cklxx/arle/tree/main/crates/cli", body: "<code>arle</code> fans out to <code>run</code>, <code>serve</code>, <code>train</code>, and <code>data</code> verbs. One stable CLI instead of a scatter of task binaries." },
        { cap: "runtime spine", href: "https://github.com/cklxx/arle/tree/main/infer/src", body: "<code>infer</code> owns scheduling, model loading, ops, backend dispatch, and the OpenAI-compatible HTTP surface." },
        { cap: "kernel crates", href: "https://github.com/cklxx/arle/tree/main/crates", body: "<code>cuda-kernels</code> ships the CUDA kernel + Triton AOT prelude. <code>mlx-sys</code> is the single Apple Silicon C++ bridge. <code>kv-native-sys</code> is the local KV-tier substrate." },
      ],
    },
    surfaces: {
      title: "SURFACES",
      rows: [
        { cap: "<code>infer</code>", body: "OpenAI-compatible HTTP serving, metrics, stats, and session endpoints on the runtime's canonical server surface." },
        { cap: "<code>arle run</code>", body: "Interactive REPL and one-shot prompt execution with the same Rust runtime authority behind the local agent loop." },
        { cap: "<code>arle serve</code>", body: "Unified front door that launches the matching serving binary from the release artifact or PATH." },
        { cap: "<code>arle train</code>", body: "Pretrain, SFT, GRPO, multi-turn RL, and eval workflows through one front door instead of scattered train binaries." },
        { cap: "<code>arle data</code>", body: "Dataset download and schema conversion utilities that stay versioned with the same workspace and docs." },
      ],
    },
    matrix: {
      title: "SUPPORT MATRIX",
      caption:
        "Three backends, one runtime contract. Authoritative truth lives in <a href=\"https://github.com/cklxx/arle/blob/main/docs/support-matrix.md\">docs/support-matrix.md</a>.",
      head: ["backend", "stability", "os / hardware", "models", "quants", "api"],
      rows: [
        [
          "<code>cuda</code>",
          '<span class="m-ok">stable</span>',
          "Linux + NVIDIA Ampere+",
          "Qwen3 / Qwen3.5",
          "FP16 / BF16, GGUF Q4_K",
          "OpenAI v1",
        ],
        [
          "<code>metal</code>",
          '<span class="m-warn">beta</span>',
          "Apple Silicon (M1+)",
          "Qwen3 / Qwen3.5",
          "FP16 / BF16, dense GGUF",
          "OpenAI v1",
        ],
        [
          "<code>cpu</code>",
          '<span class="m-dim">dev only</span>',
          "portable smoke",
          "Qwen3 / Qwen3.5 (small)",
          "FP16 / BF16",
          "OpenAI v1",
        ],
      ],
      note:
        'Stable means CI-gated and shipped; beta means actively validated but uneven; dev-only is for smoke coverage on machines without a GPU.',
    },
    status: {
      title: "STATUS",
      rows: [
        { cap: "Project posture", body: "Runtime-first. <code>infer</code> is the primary serving surface; <code>arle</code> extends the same runtime into agent, train, eval, and data flows." },
        { cap: "Backends", body: "CUDA stable on Linux + NVIDIA Ampere+. Metal beta on Apple Silicon. CPU dev-only for smoke." },
        { cap: "Models", body: "Qwen3 and Qwen3.5 ship today. Llama 3 / 4 and DeepSeek V3 / R1 remain planned, not implied." },
        { cap: "HTTP", body: "<code>/v1/completions</code>, <code>/v1/chat/completions</code>, <code>/v1/models</code>, <code>/metrics</code>, <code>/v1/stats</code> — stable public surface." },
        { cap: "Bench program", body: 'Dated snapshots live in <a href="https://github.com/cklxx/arle/tree/main/docs/experience/wins">docs/experience/wins/</a>. Tooling: <code>scripts/bench_guidellm.sh</code>, locked in <a href="https://github.com/cklxx/arle/blob/main/docs/plans/guidellm-integration.md">guidellm-integration.md</a>.' },
      ],
    },
    bench: {
      title: "BENCH",
      v2Heading: "Real numbers. Dated, reproducible, in-repo.",
      caption:
        'Dated snapshots straight from <a href="https://github.com/cklxx/arle/tree/main/docs/experience/wins">docs/experience/wins/</a>. Numbers come out of <code>scripts/bench_guidellm.sh</code> and the canonical step-driver smokes &mdash; nothing is curated.',
      head: ["date", "backend", "model", "hardware", "metric", ""],
      rows: [
        {
          date: "2026-04-23",
          backend: "cuda",
          model: "Qwen3-4B",
          hardware: "NVIDIA L4 · BF16",
          metric: 'ITL p50 <b>59.93 ms</b> &middot; out <b>118 tok/s</b> &middot; conc=16',
          href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-23-bench-guidellm-qwen3-4b-l4-c16-tier-prefetch-42ce889.md",
          stability: "stable",
          cmd: "infer bench --model Qwen3-4B",
          grid: [
            {
              label: "output throughput",
              value: "118",
              unit: "tok/s",
              spark: "M0 12 L10 11 L20 13 L30 9 L40 8 L50 6 L64 4",
            },
            {
              label: "ITL p50",
              value: "59.9",
              unit: "ms",
              spark: "M0 6 L10 7 L20 6 L30 9 L40 8 L50 11 L64 13",
            },
            { label: "concurrency", value: "16", unit: "streams" },
            { label: "first token", value: "412", unit: "ms" },
          ],
        },
        {
          date: "2026-04-27",
          backend: "metal",
          model: "Qwen3.5-0.8B Q4_K_M",
          hardware: "Apple M4 Pro · GGUF decode",
          metric: 'gen <b>211 tok/s</b> &middot; e2e <b>202 tok/s</b> &middot; gguf affine repack',
          href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md",
          stability: "beta",
          cmd: "metal_bench --model Qwen3.5-0.8B-Q4_K_M.gguf",
          grid: [
            {
              label: "gen tok/s",
              value: "211",
              unit: "tok/s",
              spark: "M0 14 L10 11 L20 9 L30 7 L40 5 L50 4 L64 3",
            },
            {
              label: "decode ms/tok",
              value: "4.7",
              unit: "ms",
              spark: "M0 12 L10 10 L20 9 L30 7 L40 6 L50 5 L64 4",
            },
            { label: "peak RSS", value: "1.4", unit: "GiB" },
            { label: "TTFT", value: "223", unit: "ms" },
          ],
        },
      ],
      note: 'See full snapshots for env, params, regression deltas, and the problems the bench surfaced. CUDA closure work currently lives in <a href="https://github.com/cklxx/arle/blob/main/docs/plans/2026-04-23-cuda-decode-sglang-alignment.md">the decode-alignment plan</a>.',
    },
    quickstart: {
      title: "QUICKSTART",
      cards: [
        {
          title: "— CUDA container",
          lines: [
            '<span class="p">$</span> docker run --rm --gpus all -p 8000:8000 \\',
            '    -v /path/to/Qwen3-4B:/model:ro \\',
            '    ghcr.io/cklxx/arle:latest \\',
            '    serve --backend cuda --model-path /model --port 8000',
          ],
        },
        {
          title: "— Local CLI smoke",
          lines: [
            '<span class="p">$</span> git clone https://github.com/cklxx/arle',
            '<span class="p">$</span> cd arle',
            '<span class="p">$</span> ./setup.sh',
            '<span class="p">$</span> cargo build --release --no-default-features \\',
            '    --features cpu,no-cuda,cli --bin arle',
            '<span class="p">$</span> ./target/release/arle --doctor',
            '<span class="p">$</span> ./target/release/arle --model-path /path/to/Qwen3-0.6B \\',
            '    run --no-tools --prompt <i>"Say hello in one sentence"</i>',
            '<span class="p">$</span> ./target/release/arle train env',
          ],
        },
        {
          title: "— Serving smoke",
          lines: [
            '<span class="p">$</span> cargo build -p infer --release',
            '<span class="p">$</span> cargo build --release --features cli --bin arle',
            '<span class="p">$</span> ./target/release/arle serve --backend cuda \\',
            '    --model-path /path/to/Qwen3-4B --port 8000',
            '<span class="p">$</span> curl http://127.0.0.1:8000/v1/chat/completions \\',
            '    -H <i>\'Content-Type: application/json\'</i> \\',
            '    -d <i>\'{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}\'</i>',
          ],
        },
      ],
      note:
        'Use <code>--features cli</code> for the default CUDA build, <code>metal,no-cuda,cli</code> on Apple Silicon, and <code>cpu,no-cuda,cli</code> for portable smoke paths. <code>./setup.sh</code> bootstraps Rust, Python, Zig, and local checks.',
    },
    examples: {
      title: "EXAMPLES",
      lead:
        "An OpenAI-compatible chat call against a local <code>infer</code> server, piped through <code>jq</code> for the assistant message:",
      lines: [
        '<span class="ln">1</span><span class="p">$</span> curl -s http://127.0.0.1:8000/v1/chat/completions \\',
        '<span class="ln"> </span>    -H <i>\'Content-Type: application/json\'</i> \\',
        '<span class="ln"> </span>    -d <i>\'{"model":"qwen3-4b","messages":[{"role":"user","content":"Define ARLE in 8 words."}],"max_tokens":40}\'</i> \\',
        '<span class="ln"> </span>    | jq .choices[0].message',
        '<span class="ln">2</span>{',
        '<span class="ln"> </span>  "role": <i>"assistant"</i>,',
        '<span class="ln"> </span>  "content": <i>"runtime-first Rust workspace serving and orchestrating local agents."</i>',
        '<span class="ln"> </span>}',
      ],
      seeAlso:
        'See also <a href="#synopsis">SYNOPSIS</a>, <a href="#surfaces">SURFACES</a>, <a href="#quickstart">QUICKSTART</a>.',
    },
    files: {
      title: "FILES",
      rows: FILES_EN,
    },
    seealso: {
      title: "SEE ALSO",
      items: [
        '<a href="https://github.com/cklxx/arle/blob/main/CHANGELOG.md">CHANGELOG.md</a> — dated project history',
        '<a href="https://github.com/cklxx/arle/blob/main/ROADMAP.md">ROADMAP.md</a> — next milestones',
        '<a href="https://github.com/cklxx/arle/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a> — contributor setup and verification gates',
        '<a href="https://github.com/cklxx/arle/blob/main/SECURITY.md">SECURITY.md</a> — private vulnerability reporting policy',
      ],
      note:
        'If you are here to contribute, start with <a href="https://github.com/cklxx/arle/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>, not the maintainer plans tree.',
    },
  },
  docfoot: {
    cols: DOCFOOT_EN,
    meta:
      'Documentation site coming. For now everything stable lives in the <a href="https://github.com/cklxx/arle">repo</a>.',
  },
};

export const ZH: Locale = {
  lang: "zh-Hans",
  hreflang: "zh-Hans",
  meta: {
    title: "ARLE(1) — runtime-first Rust workspace",
    description:
      "ARLE 是以 runtime 为主干的 Rust workspace，覆盖 serving、本地 agent、训练、评测与数据集工具。",
    ogTitle: "ARLE — runtime-first Rust workspace",
    ogDescription:
      "同一套 Rust workspace 里收口 infer serving、arle 前门、训练 / 评测、数据集工具与公共文档。",
    ogUrl: "https://cklxx.github.io/arle/zh-cn/",
    canonical: "https://cklxx.github.io/arle/zh-cn/",
  },
  termtabs: { on: "man arle.1", cwd: "~/code/arle", meta: "zsh · 104×42" },
  manhead: { left: "ARLE(1)", center: "用户命令手册", right: "ARLE(1)" },
  manfoot: { left: "ARLE(1)", center: "2026 年 4 月", right: "ARLE(1)" },
  hero: {
    kicker: "runtime-first rust workspace",
    tagline:
      '<span class="hl">infer</span> 负责 OpenAI 兼容 serving。<span class="hl">arle</span> 负责本地 agent、训练、评测、数据流。一套 Rust workspace。',
    signals: SIGNALS,
    primaryCta: "Quickstart",
    secondaryCta: "cklxx/arle",
    version: "v0.4.2",
    nav: [
      { label: "文档", href: "../docs/" },
      { label: "基准", href: "#bench" },
      { label: "安装", href: "#install" },
      { label: "示例", href: "#examples" },
    ],
    display: {
      status: "v0.4.2 stable",
      h1Lines: [
        '让 <span class="strike"><span class="txt">Qwen3</span></span>',
        "跑得跟",
        "<em>cargo build</em> 一样快。",
      ],
      replPrimary: "纯 Rust · OpenAI v1",
      replSuffix: "— CUDA + Metal + CPU",
      lede:
        '<b>infer</b> 在 CUDA、Metal、CPU 上提供 OpenAI 兼容 serving。<b>arle</b> 是 run / serve / train / data 的统一前门 — <a class="inline" href="#topology">一套 Rust workspace，零胶水。</a>',
    },
    installLabel: "install · macos · linux",
    stats: [
      {
        lbl: "cuda · L4 · qwen3-4b",
        num: "118",
        unit: "tok/s",
        meta: 'conc=<b>16</b> · BF16',
        spark: "M0 9 L8 8 L16 10 L24 7 L32 6 L40 7 L48 4 L60 3",
      },
      {
        lbl: "cuda · L4 · ITL p50",
        num: "59.9",
        unit: "ms",
        meta: '<b>2026-04-23</b>',
        spark: "M0 5 L8 6 L16 5 L24 8 L32 7 L40 9 L48 10 L60 11",
      },
      {
        lbl: "metal · M4 Pro · 0.8b q4_k_m",
        num: "211",
        unit: "tok/s",
        meta: 'gguf decode · <b>2026-04-27</b>',
        spark: "M0 11 L8 9 L16 7 L24 6 L32 5 L40 4 L48 3 L60 2",
      },
      {
        lbl: "冷启动 · arle serve",
        num: "1.4",
        unit: "s",
        meta: "加载 + 首 token",
        spark: "M0 5 L8 6 L16 5 L24 5 L32 4 L40 5 L48 4 L60 4",
      },
    ],
  },
  jumps: [
    { label: "[安装]", href: "#install" },
    { label: "[概览]", href: "#glance" },
    { label: "[拓扑]", href: "#topology" },
    { label: "[支持矩阵]", href: "#matrix" },
    { label: "[基准]", href: "#bench" },
    { label: "[快速开始]", href: "#quickstart" },
    { label: "[概要]", href: "#synopsis" },
    { label: "[示例]", href: "#examples" },
    { label: "[文档]", href: "#docs" },
    { label: "[github]", href: "https://github.com/cklxx/arle" },
    { label: "[发版]", href: "https://github.com/cklxx/arle/releases" },
  ],
  langSwitch: {
    other: { label: "en", href: "", hreflang: "en" }, // base URL is the EN page
    selfLabel: "zh",
  },
  sections: {
    name: {
      title: "名称",
      body:
        "<b>arle</b> — 以 runtime 为主干的 Rust workspace。<code>infer</code> 负责 serving，<code>arle</code> 负责操作。",
      note:
        "这是一张稳定入口；产品真相在下面列出的仓库文档里。",
    },
    install: {
      title: "安装",
      caption:
        '每个平台一行能跑的命令。完整支持矩阵与环境变量见 <a href="https://github.com/cklxx/arle/blob/main/docs/install.md">docs/install.md</a>。',
      cards: [
        {
          label: "Apple Silicon · Homebrew",
          lines: [
            '<span class="p">$</span> brew install cklxx/tap/arle',
            '<span class="p">$</span> arle --doctor',
          ],
        },
        {
          label: "Apple Silicon / Linux x86_64 · curl",
          lines: [
            '<span class="p">$</span> curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh \\',
            '    | sh',
            '<span class="p">$</span> arle --doctor',
          ],
        },
        {
          label: "CUDA · GPU 容器",
          lines: [
            '<span class="p">$</span> docker run --rm --gpus all -p 8000:8000 \\',
            '    -v /path/to/Qwen3-4B:/model:ro \\',
            '    ghcr.io/cklxx/arle:latest \\',
            '    serve --backend cuda --model-path /model',
          ],
        },
        {
          label: "源码 · Linux / Apple Silicon",
          lines: [
            '<span class="p">$</span> git clone https://github.com/cklxx/arle &amp;&amp; cd arle',
            '<span class="p">$</span> ./setup.sh',
            '<span class="p">$</span> cargo build --release --features cli --bin arle',
            '<span class="p">$</span> ./target/release/arle serve --model-path /path/to/Qwen3-4B --port 8000',
          ],
        },
      ],
      note:
        '预编译 tarball 与 SHA 都在每次 <a href="https://github.com/cklxx/arle/releases">GitHub Release</a>;curl 安装脚本会先校验 SHA256 再解压。Apple Silicon 源码构建把 <code>--features cli</code> 换成 <code>--features metal,no-cuda,cli</code>。',
    },
    glance: {
      title: "概览",
      cards: [
        {
          h: "— Runtime 主干",
          body:
            "Serving 仍是主线。<code>infer</code> 是生成、metrics、stats 与 session 的标准 HTTP 服务面。",
        },
        {
          h: "— 一张前门",
          body:
            "<code>arle run</code>、<code>arle serve</code>、<code>arle train</code> 与 <code>arle data</code> 把用户入口收成一张前门，不再分散成多组二进制。",
        },
        {
          h: "— 同一套权威",
          body:
            "运行时代码、模型加载、train/eval 流程与公共文档都落在同一套 Rust workspace 里，避免口径漂移。",
        },
      ],
    },
    synopsis: {
      title: "概要",
      lines: [
        '<span class="ln">1</span><span class="p">$</span> cargo build -p infer --release',
        '<span class="ln">2</span><span class="p">$</span> ./target/release/infer --model-path /path/to/Qwen3-4B --port 8000',
        '<span class="ln">3</span><span class="p">$</span> cargo build --release --features cli --bin arle',
        '<span class="ln">4</span><span class="p">$</span> ./target/release/arle --doctor',
        '<span class="ln">5</span><span class="p">$</span> ./target/release/arle --model-path /path/to/Qwen3-4B \\',
        '<span class="ln"> </span>    run --prompt <i>"总结一下这个仓库"</i>',
        '<span class="ln">6</span><span class="p">$</span> ./target/release/arle serve --backend cuda \\',
        '<span class="ln"> </span>    --model-path /path/to/Qwen3-4B --port 8000',
        '<span class="ln">7</span><span class="p">$</span> ./target/release/arle train env',
        '<span class="ln">8</span><span class="p">$</span> ./target/release/arle data convert --help<span class="caret"></span>',
      ],
    },
    topology: {
      title: "拓扑",
      legend: [
        { cap: "前门", href: "https://github.com/cklxx/arle/tree/main/crates/cli", body: "<code>arle</code> 通过 <code>run</code>、<code>serve</code>、<code>train</code>、<code>data</code> 几个动词分流；不再为每个任务拉独立二进制。" },
        { cap: "运行时主干", href: "https://github.com/cklxx/arle/tree/main/infer/src", body: "<code>infer</code> 负责调度、模型加载、ops、后端 dispatch 与 OpenAI 兼容 HTTP 服务面。" },
        { cap: "Kernel 子 crate", href: "https://github.com/cklxx/arle/tree/main/crates", body: "<code>cuda-kernels</code> 汇集 CUDA kernel 与 Triton AOT prelude。<code>mlx-sys</code> 是 Apple Silicon 唯一的 C++ 桥。<code>kv-native-sys</code> 是本地 KV-tier 基座。" },
      ],
    },
    surfaces: {
      title: "入口面",
      rows: [
        { cap: "<code>infer</code>", body: "OpenAI 兼容 HTTP serving、metrics、stats 与 session 接口都收口在运行时的标准服务面上。" },
        { cap: "<code>arle run</code>", body: "交互式 REPL 与一次性 prompt 执行，共用同一套 Rust 运行时权威，直接面向本地 agent 工作流。" },
        { cap: "<code>arle serve</code>", body: "统一前门，从 release artifact 或 PATH 拉起匹配的 serving 二进制。" },
        { cap: "<code>arle train</code>", body: "Pretrain、SFT、GRPO、多轮 RL 与 eval 共用一张前门，不再依赖分散的训练二进制。" },
        { cap: "<code>arle data</code>", body: "数据集下载与 schema 转换工具，和同一套 workspace、版本、文档一起演进。" },
      ],
    },
    matrix: {
      title: "支持矩阵",
      caption:
        '三种后端，一份运行时契约。权威矩阵见 <a href="https://github.com/cklxx/arle/blob/main/docs/support-matrix.md">docs/support-matrix.md</a>。',
      head: ["后端", "稳定度", "系统 / 硬件", "模型", "量化", "API"],
      rows: [
        [
          "<code>cuda</code>",
          '<span class="m-ok">stable</span>',
          "Linux + NVIDIA Ampere+",
          "Qwen3 / Qwen3.5",
          "FP16 / BF16、GGUF Q4_K",
          "OpenAI v1",
        ],
        [
          "<code>metal</code>",
          '<span class="m-warn">beta</span>',
          "Apple Silicon（M1+）",
          "Qwen3 / Qwen3.5",
          "FP16 / BF16、dense GGUF",
          "OpenAI v1",
        ],
        [
          "<code>cpu</code>",
          '<span class="m-dim">dev only</span>',
          "便携冒烟",
          "Qwen3 / Qwen3.5（小尺寸）",
          "FP16 / BF16",
          "OpenAI v1",
        ],
      ],
      note: "stable = 已上 CI 与发版；beta = 持续验证但稳定性不齐；dev-only = 仅用于无 GPU 机器上的冒烟覆盖。",
    },
    status: {
      title: "状态",
      rows: [
        { cap: "项目姿态", body: "Runtime-first。<code>infer</code> 是主 serving 面；<code>arle</code> 在同一套运行时之上向 agent、train、eval、data 扩展。" },
        { cap: "后端", body: "CUDA 在 Linux + NVIDIA Ampere+ 上 stable。Metal 在 Apple Silicon 上 beta。CPU 仅用于冒烟。" },
        { cap: "模型", body: "已交付 Qwen3 与 Qwen3.5。Llama 3 / 4 与 DeepSeek V3 / R1 仍是规划，不做暗示。" },
        { cap: "HTTP", body: "<code>/v1/completions</code>、<code>/v1/chat/completions</code>、<code>/v1/models</code>、<code>/metrics</code>、<code>/v1/stats</code> — 稳定公共服务面。" },
        { cap: "Bench 程序", body: '带日期快照在 <a href="https://github.com/cklxx/arle/tree/main/docs/experience/wins">docs/experience/wins/</a>。工具：<code>scripts/bench_guidellm.sh</code>，参数锁在 <a href="https://github.com/cklxx/arle/blob/main/docs/plans/guidellm-integration.md">guidellm-integration.md</a>。' },
      ],
    },
    bench: {
      title: "基准",
      v2Heading: "真实数据。带日期、可复现、在仓库里。",
      caption:
        '直接来自 <a href="https://github.com/cklxx/arle/tree/main/docs/experience/wins">docs/experience/wins/</a> 的带日期快照。数字出自 <code>scripts/bench_guidellm.sh</code> 与标准 step-driver 冒烟，未做挑选。',
      head: ["日期", "后端", "模型", "硬件", "指标", ""],
      rows: [
        {
          date: "2026-04-23",
          backend: "cuda",
          model: "Qwen3-4B",
          hardware: "NVIDIA L4 · BF16",
          metric: 'ITL p50 <b>59.93 ms</b> &middot; 输出 <b>118 tok/s</b> &middot; conc=16',
          href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-23-bench-guidellm-qwen3-4b-l4-c16-tier-prefetch-42ce889.md",
          stability: "stable",
          cmd: "infer bench --model Qwen3-4B",
          grid: [
            {
              label: "输出吞吐",
              value: "118",
              unit: "tok/s",
              spark: "M0 12 L10 11 L20 13 L30 9 L40 8 L50 6 L64 4",
            },
            {
              label: "ITL p50",
              value: "59.9",
              unit: "ms",
              spark: "M0 6 L10 7 L20 6 L30 9 L40 8 L50 11 L64 13",
            },
            { label: "并发", value: "16", unit: "streams" },
            { label: "首 token", value: "412", unit: "ms" },
          ],
        },
        {
          date: "2026-04-27",
          backend: "metal",
          model: "Qwen3.5-0.8B Q4_K_M",
          hardware: "Apple M4 Pro · GGUF decode",
          metric: 'gen <b>211 tok/s</b> &middot; e2e <b>202 tok/s</b> &middot; GGUF affine repack',
          href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md",
          stability: "beta",
          cmd: "metal_bench --model Qwen3.5-0.8B-Q4_K_M.gguf",
          grid: [
            {
              label: "生成吞吐",
              value: "211",
              unit: "tok/s",
              spark: "M0 14 L10 11 L20 9 L30 7 L40 5 L50 4 L64 3",
            },
            {
              label: "decode ms/tok",
              value: "4.7",
              unit: "ms",
              spark: "M0 12 L10 10 L20 9 L30 7 L40 6 L50 5 L64 4",
            },
            { label: "峰值内存", value: "1.4", unit: "GiB" },
            { label: "TTFT", value: "223", unit: "ms" },
          ],
        },
      ],
      note: '完整环境、参数、回归 Δ 与 bench 暴露的问题请看快照本身。当前 CUDA 收口工作见 <a href="https://github.com/cklxx/arle/blob/main/docs/plans/2026-04-23-cuda-decode-sglang-alignment.md">decode 对齐计划</a>。',
    },
    quickstart: {
      title: "快速开始",
      cards: [
        {
          title: "— CUDA 容器",
          lines: [
            '<span class="p">$</span> docker run --rm --gpus all -p 8000:8000 \\',
            '    -v /path/to/Qwen3-4B:/model:ro \\',
            '    ghcr.io/cklxx/arle:latest \\',
            '    serve --backend cuda --model-path /model --port 8000',
          ],
        },
        {
          title: "— 本地 CLI 冒烟",
          lines: [
            '<span class="p">$</span> git clone https://github.com/cklxx/arle',
            '<span class="p">$</span> cd arle',
            '<span class="p">$</span> ./setup.sh',
            '<span class="p">$</span> cargo build --release --no-default-features \\',
            '    --features cpu,no-cuda,cli --bin arle',
            '<span class="p">$</span> ./target/release/arle --doctor',
            '<span class="p">$</span> ./target/release/arle --model-path /path/to/Qwen3-0.6B \\',
            '    run --no-tools --prompt <i>"用一句话打个招呼"</i>',
            '<span class="p">$</span> ./target/release/arle train env',
          ],
        },
        {
          title: "— Serving 冒烟",
          lines: [
            '<span class="p">$</span> cargo build -p infer --release',
            '<span class="p">$</span> cargo build --release --features cli --bin arle',
            '<span class="p">$</span> ./target/release/arle serve --backend cuda \\',
            '    --model-path /path/to/Qwen3-4B --port 8000',
            '<span class="p">$</span> curl http://127.0.0.1:8000/v1/chat/completions \\',
            '    -H <i>\'Content-Type: application/json\'</i> \\',
            '    -d <i>\'{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}\'</i>',
          ],
        },
      ],
      note:
        '默认 CUDA CLI 用 <code>--features cli</code>，Apple Silicon 用 <code>metal,no-cuda,cli</code>，便携冒烟路径用 <code>cpu,no-cuda,cli</code>。<code>./setup.sh</code> 会引导 Rust、Python、Zig 和本地检查。',
    },
    examples: {
      title: "示例",
      lead:
        "对本地 <code>infer</code> server 发一次 OpenAI 兼容的 chat 请求，再用 <code>jq</code> 把 assistant 消息切出来：",
      lines: [
        '<span class="ln">1</span><span class="p">$</span> curl -s http://127.0.0.1:8000/v1/chat/completions \\',
        '<span class="ln"> </span>    -H <i>\'Content-Type: application/json\'</i> \\',
        '<span class="ln"> </span>    -d <i>\'{"model":"qwen3-4b","messages":[{"role":"user","content":"用 8 个词解释 ARLE。"}],"max_tokens":40}\'</i> \\',
        '<span class="ln"> </span>    | jq .choices[0].message',
        '<span class="ln">2</span>{',
        '<span class="ln"> </span>  "role": <i>"assistant"</i>,',
        '<span class="ln"> </span>  "content": <i>"以 runtime 为主干的 Rust workspace，统一 serving 与本地 agent。"</i>',
        '<span class="ln"> </span>}',
      ],
      seeAlso:
        '另见 <a href="#synopsis">概要</a>、<a href="#surfaces">入口面</a>、<a href="#quickstart">快速开始</a>。',
    },
    files: {
      title: "文件",
      rows: FILES_ZH,
    },
    seealso: {
      title: "另见",
      items: [
        '<a href="https://github.com/cklxx/arle/blob/main/CHANGELOG.md">CHANGELOG.md</a> — 带日期的项目历史',
        '<a href="https://github.com/cklxx/arle/blob/main/ROADMAP.md">ROADMAP.md</a> — 接下来的里程碑',
        '<a href="https://github.com/cklxx/arle/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a> — 贡献者初始化与验证门槛',
        '<a href="https://github.com/cklxx/arle/blob/main/SECURITY.md">SECURITY.md</a> — 私有漏洞报告策略',
      ],
      note:
        '如果你是来提改动的，先看 <a href="https://github.com/cklxx/arle/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>，不要先钻进维护者计划树。',
    },
  },
  docfoot: {
    cols: DOCFOOT_ZH,
    meta:
      '完整文档站点正在筹备中，目前所有稳定信息都收口在 <a href="https://github.com/cklxx/arle">repo</a>。',
  },
};
