// Content for the ARLE landing page. EN + ZH share the same component
// shapes; copy is the only thing that changes between locales.

export type TopNavLink = { label: string; href: string };

export type Signal = {
  /** Inner HTML — typically `<b>cuda</b> stable · ampere+`. */
  html: string;
  /** Status pinpoint left of the label. Omit for plain typeset signals. */
  dot?: "ok" | "warn" | "dim";
};

export type CtaLink = { label: string; href: string };

export type Terminal = {
  /** Title shown in the dark macOS-bar of the terminal block. */
  title: string;
  /** Right-hand cwd in the bar. */
  cwd: string;
  /** Lines of the <pre> body. Raw HTML allowed (use <span class="p|c|ok|warn|k|out|dim">). */
  lines: string[];
};

export type InstallCard = {
  /** Bold label on the left of the card header (e.g. "Apple Silicon · Homebrew"). */
  label: string;
  /** Right-hand suffix on the card header (e.g. "zsh / bash"). */
  channel: string;
  /** <pre> body. Raw HTML allowed. */
  lines: string[];
};

export type BenchCell = {
  /** Uppercase key above the metric (e.g. "output"). */
  key: string;
  /** Numeric value (e.g. "118"). */
  value: string;
  /** Short unit (e.g. "tok/s"). */
  unit: string;
};

export type BenchRow = {
  /** ISO date prefixed at the top of the card header. */
  date: string;
  /** Stability tag rendered in the card header. */
  stability: "stable" | "beta" | "dev";
  /** Visible label of the stability tag (e.g. "stable · ci-gated"). */
  stabilityLabel: string;
  /** Caption under the header — backend, hardware, model. Raw HTML allowed. */
  caption: string;
  /** Four metric cells in a 2×2 grid. */
  cells: BenchCell[];
  /** Bottom-left command. */
  cmd: string;
  /** Bottom-right snapshot link. */
  href: string;
};

export type Matrix = {
  caption: string; // raw HTML allowed
  head: string[];
  rows: string[][]; // raw HTML allowed per cell
};

export type Install = {
  caption: string; // raw HTML allowed
  cards: InstallCard[];
};

export type Bench = {
  caption: string; // raw HTML allowed
  rows: BenchRow[];
};

export type FileRow = {
  path: string;
  desc: string;
  href: string;
};

export type Files = {
  caption: string; // raw HTML allowed
  rows: FileRow[];
};

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
  /** Top masthead — `arle(1)` left, uppercase nav right. */
  masthead: {
    left: string;
    links: TopNavLink[];
  };
  hero: {
    /** Wordmark text after the coral period — almost always "arle". */
    wordmark: string;
    /** Lede paragraph. Raw HTML allowed. */
    lede: string;
    signals: Signal[];
    primaryCta: CtaLink;
    secondaryCta: CtaLink;
    terminal: Terminal;
  };
  sections: {
    install: { title: string } & Install;
    bench: { title: string } & Bench;
    matrix: { title: string } & Matrix;
    files: { title: string } & Files;
  };
  footer: {
    /** Plain-text left meta (e.g. "arle(1) · April 2026 · v0.4.2"). */
    left: string;
    /** Right meta as a link. */
    right: { label: string; href: string };
  };
};

const SIGNALS: Signal[] = [
  { html: '<b>cuda</b> stable · ampere+', dot: "ok" },
  { html: '<b>metal</b> beta · apple silicon', dot: "warn" },
  { html: '<b>cpu</b> dev only', dot: "dim" },
  { html: '<b>api</b> openai · v1' },
  { html: '<b>release</b> v0.4.2 · 2026-04' },
];

const TERMINAL_LINES_EN = [
  '<span class="p">$</span> arle --doctor',
  '<span class="out">cuda    </span><span class="ok">ok</span>    <span class="c"># nvidia-smi · cuda 12.x · ampere+</span>',
  '<span class="out">metal   </span><span class="warn">beta</span>  <span class="c"># apple m-series detected</span>',
  '<span class="out">cpu     </span><span class="ok">ok</span>    <span class="c"># dev-only smoke path</span>',
  '<span class="out">model   </span><span class="ok">ok</span>    <span class="c"># Qwen3-4B reachable</span>',
  '<span class="out">api     </span><span class="ok">ok</span>    <span class="c"># /v1/chat/completions · streaming</span>',
  "",
  '<span class="p">$</span> arle <span class="k">serve</span> --backend cuda --model Qwen3-4B',
  '<span class="dim">listening on</span> http://0.0.0.0:8000  <span class="dim">·</span> <span class="ok">ready</span> <span class="dim">in 1.4s</span>',
];

const INSTALL_CARDS_EN: InstallCard[] = [
  {
    label: "Apple Silicon · Homebrew",
    channel: "zsh / bash",
    lines: [
      '<span class="p">$</span> brew install cklxx/tap/arle',
      '<span class="p">$</span> arle --doctor',
    ],
  },
  {
    label: "Linux x86_64 / macOS · curl",
    channel: "sh-compatible",
    lines: [
      '<span class="p">$</span> curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh \\',
      '    | sh',
      '<span class="p">$</span> arle --doctor',
    ],
  },
  {
    label: "CUDA · GPU container",
    channel: "docker / nvidia",
    lines: [
      '<span class="p">$</span> docker run --rm --gpus all -p 8000:8000 \\',
      '    -v /path/to/Qwen3-4B:/model:ro \\',
      '    ghcr.io/cklxx/arle:latest \\',
      '    serve --backend cuda --model-path /model',
    ],
  },
  {
    label: "Source · Cargo",
    channel: "workspace",
    lines: [
      '<span class="p">$</span> git clone https://github.com/cklxx/arle &amp;&amp; cd arle',
      '<span class="p">$</span> cargo install --path crates/cli --features cuda',
      '<span class="c"># --features cuda is opt-in; cpu builds out of the box</span>',
    ],
  },
];

const INSTALL_CARDS_ZH: InstallCard[] = [
  {
    label: "Apple Silicon · Homebrew",
    channel: "zsh / bash",
    lines: [
      '<span class="p">$</span> brew install cklxx/tap/arle',
      '<span class="p">$</span> arle --doctor',
    ],
  },
  {
    label: "Linux x86_64 / macOS · curl",
    channel: "sh 兼容",
    lines: [
      '<span class="p">$</span> curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh \\',
      '    | sh',
      '<span class="p">$</span> arle --doctor',
    ],
  },
  {
    label: "CUDA · GPU 容器",
    channel: "docker / nvidia",
    lines: [
      '<span class="p">$</span> docker run --rm --gpus all -p 8000:8000 \\',
      '    -v /path/to/Qwen3-4B:/model:ro \\',
      '    ghcr.io/cklxx/arle:latest \\',
      '    serve --backend cuda --model-path /model',
    ],
  },
  {
    label: "源码 · Cargo",
    channel: "workspace",
    lines: [
      '<span class="p">$</span> git clone https://github.com/cklxx/arle &amp;&amp; cd arle',
      '<span class="p">$</span> cargo install --path crates/cli --features cuda',
      '<span class="c"># --features cuda 可选; cpu 默认就能编</span>',
    ],
  },
];

const BENCH_ROWS_EN: BenchRow[] = [
  {
    date: "2026-04-23",
    stability: "stable",
    stabilityLabel: "stable · ci-gated",
    caption: '<b>cuda</b> · NVIDIA L4 · <code>Qwen3-4B</code> · BF16 · concurrency=16',
    cells: [
      { key: "output", value: "118", unit: "tok/s" },
      { key: "itl p50", value: "59.9", unit: "ms" },
      { key: "first token", value: "412", unit: "ms" },
      { key: "streams", value: "16", unit: "conc" },
    ],
    cmd: "infer bench --model Qwen3-4B",
    href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-23-bench-guidellm-qwen3-4b-l4-c16-tier-prefetch-42ce889.md",
  },
  {
    date: "2026-04-27",
    stability: "beta",
    stabilityLabel: "beta · validated",
    caption: '<b>metal</b> · Apple M4 Pro · <code>Qwen3.5-0.8B Q4_K_M</code> · GGUF decode',
    cells: [
      { key: "gen", value: "211", unit: "tok/s" },
      { key: "e2e", value: "202", unit: "tok/s" },
      { key: "decode", value: "4.7", unit: "ms/tok" },
      { key: "ttft", value: "223", unit: "ms" },
    ],
    cmd: "metal_bench --model Qwen3.5-0.8B-Q4_K_M.gguf",
    href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md",
  },
];

const BENCH_ROWS_ZH: BenchRow[] = [
  {
    date: "2026-04-23",
    stability: "stable",
    stabilityLabel: "stable · CI 已门控",
    caption: '<b>cuda</b> · NVIDIA L4 · <code>Qwen3-4B</code> · BF16 · concurrency=16',
    cells: [
      { key: "输出", value: "118", unit: "tok/s" },
      { key: "ITL p50", value: "59.9", unit: "ms" },
      { key: "首 token", value: "412", unit: "ms" },
      { key: "并发", value: "16", unit: "conc" },
    ],
    cmd: "infer bench --model Qwen3-4B",
    href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-23-bench-guidellm-qwen3-4b-l4-c16-tier-prefetch-42ce889.md",
  },
  {
    date: "2026-04-27",
    stability: "beta",
    stabilityLabel: "beta · 持续验证",
    caption: '<b>metal</b> · Apple M4 Pro · <code>Qwen3.5-0.8B Q4_K_M</code> · GGUF decode',
    cells: [
      { key: "生成", value: "211", unit: "tok/s" },
      { key: "e2e", value: "202", unit: "tok/s" },
      { key: "decode", value: "4.7", unit: "ms/tok" },
      { key: "TTFT", value: "223", unit: "ms" },
    ],
    cmd: "metal_bench --model Qwen3.5-0.8B-Q4_K_M.gguf",
    href: "https://github.com/cklxx/arle/blob/main/docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md",
  },
];

const MATRIX_ROWS_EN: string[][] = [
  [
    "<code>cuda</code>",
    '<span class="pill ok">stable</span>',
    "Linux + NVIDIA Ampere+",
    "Qwen3 / Qwen3.5",
    "FP16 / BF16, GGUF Q4_K",
    "OpenAI v1",
  ],
  [
    "<code>metal</code>",
    '<span class="pill warn">beta</span>',
    "Apple Silicon (M1+)",
    "Qwen3 / Qwen3.5",
    "FP16 / BF16, dense GGUF",
    "OpenAI v1",
  ],
  [
    "<code>cpu</code>",
    '<span class="pill dim">dev only</span>',
    "portable smoke",
    "Qwen3 / Qwen3.5 (small)",
    "FP16 / BF16",
    "OpenAI v1",
  ],
];

const MATRIX_ROWS_ZH: string[][] = [
  [
    "<code>cuda</code>",
    '<span class="pill ok">stable</span>',
    "Linux + NVIDIA Ampere+",
    "Qwen3 / Qwen3.5",
    "FP16 / BF16、GGUF Q4_K",
    "OpenAI v1",
  ],
  [
    "<code>metal</code>",
    '<span class="pill warn">beta</span>',
    "Apple Silicon（M1+）",
    "Qwen3 / Qwen3.5",
    "FP16 / BF16、dense GGUF",
    "OpenAI v1",
  ],
  [
    "<code>cpu</code>",
    '<span class="pill dim">dev only</span>',
    "便携冒烟",
    "Qwen3 / Qwen3.5（小尺寸）",
    "FP16 / BF16",
    "OpenAI v1",
  ],
];

const FILES_EN: FileRow[] = [
  { path: "/README.md", desc: "public overview · install · CLI · architecture", href: "https://github.com/cklxx/arle/blob/main/README.md" },
  { path: "/docs/http-api.md", desc: "HTTP contract · streaming behavior", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
  { path: "/docs/support-matrix.md", desc: "backend / model / quant support", href: "https://github.com/cklxx/arle/blob/main/docs/support-matrix.md" },
  { path: "/docs/stability-policy.md", desc: "stability levels · compatibility posture", href: "https://github.com/cklxx/arle/blob/main/docs/stability-policy.md" },
  { path: "/docs/experience/wins/", desc: "dated benchmark snapshots", href: "https://github.com/cklxx/arle/tree/main/docs/experience/wins" },
  { path: "/crates/cli/", desc: "arle binary · verbs · doctor", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
  { path: "/infer/", desc: "runtime spine · scheduler · loader · http", href: "https://github.com/cklxx/arle/tree/main/infer" },
  { path: "/crates/cuda-kernels/", desc: "cuda kernel crate · csrc · prelude", href: "https://github.com/cklxx/arle/tree/main/crates/cuda-kernels" },
  { path: "/crates/mlx-sys/", desc: "metal bridge · cmake + cc", href: "https://github.com/cklxx/arle/tree/main/crates/mlx-sys" },
  { path: "/examples/", desc: "copyable curl · Docker · Metal · tiny train smokes", href: "https://github.com/cklxx/arle/tree/main/examples" },
  { path: "/releases", desc: "tagged binaries · checksums", href: "https://github.com/cklxx/arle/releases" },
];

const FILES_ZH: FileRow[] = [
  { path: "/README.zh-CN.md", desc: "中文公共入口：安装 · CLI · 架构", href: "https://github.com/cklxx/arle/blob/main/README.zh-CN.md" },
  { path: "/docs/http-api.md", desc: "HTTP 契约 · 流式行为", href: "https://github.com/cklxx/arle/blob/main/docs/http-api.md" },
  { path: "/docs/support-matrix.md", desc: "后端 / 模型 / 量化支持", href: "https://github.com/cklxx/arle/blob/main/docs/support-matrix.md" },
  { path: "/docs/stability-policy.md", desc: "稳定性分级 · 兼容性姿态", href: "https://github.com/cklxx/arle/blob/main/docs/stability-policy.md" },
  { path: "/docs/experience/wins/", desc: "带日期的基准快照", href: "https://github.com/cklxx/arle/tree/main/docs/experience/wins" },
  { path: "/crates/cli/", desc: "arle 二进制 · 子命令 · doctor", href: "https://github.com/cklxx/arle/tree/main/crates/cli" },
  { path: "/infer/", desc: "运行时主干 · scheduler · loader · http", href: "https://github.com/cklxx/arle/tree/main/infer" },
  { path: "/crates/cuda-kernels/", desc: "cuda kernel crate · csrc · prelude", href: "https://github.com/cklxx/arle/tree/main/crates/cuda-kernels" },
  { path: "/crates/mlx-sys/", desc: "metal 桥接 · cmake + cc", href: "https://github.com/cklxx/arle/tree/main/crates/mlx-sys" },
  { path: "/examples/", desc: "curl · Docker · Metal · tiny train 冒烟示例", href: "https://github.com/cklxx/arle/tree/main/examples" },
  { path: "/releases", desc: "发版二进制 · 校验和", href: "https://github.com/cklxx/arle/releases" },
];

export const EN: Locale = {
  lang: "en",
  hreflang: "en",
  meta: {
    title: "arle(1) — runtime-first rust workspace",
    description:
      "ARLE is a runtime-first Rust workspace for serving Qwen3/Qwen3.5 on CUDA, Metal, and CPU. infer serves OpenAI-compatible traffic; arle is the unified front door for run, serve, train, and data flows.",
    ogTitle: "arle — runtime-first Rust workspace",
    ogDescription:
      "infer serves OpenAI-compatible traffic on CUDA, Metal, and CPU. arle is the unified front door for run, serve, train, and data.",
    ogUrl: "https://cklxx.github.io/arle/",
    canonical: "https://cklxx.github.io/arle/",
  },
  masthead: {
    left: "arle(1)",
    links: [
      { label: "install", href: "#install" },
      { label: "bench", href: "#bench" },
      { label: "matrix", href: "#matrix" },
      { label: "github ↗", href: "https://github.com/cklxx/arle" },
    ],
  },
  hero: {
    wordmark: "arle",
    lede:
      "A runtime-first Rust workspace. <b>infer</b> serves OpenAI-compatible traffic on CUDA, Metal, and CPU; <b>arle</b> is the unified front door for run, serve, train, and data flows.",
    signals: SIGNALS,
    primaryCta: { label: "$ Quickstart", href: "#install" },
    secondaryCta: { label: "cklxx/arle ↗", href: "https://github.com/cklxx/arle" },
    terminal: {
      title: "arle — bash",
      cwd: "~/projects/arle",
      lines: TERMINAL_LINES_EN,
    },
  },
  sections: {
    install: {
      title: "Install",
      caption:
        'One runnable line per platform. Pre-built tarballs and SHAs on each <a href="https://github.com/cklxx/arle/releases">GitHub Release</a>; the curl installer verifies SHA256 before extracting.',
      cards: INSTALL_CARDS_EN,
    },
    bench: {
      title: "Bench",
      caption:
        'Dated, reproducible snapshots straight from <a href="https://github.com/cklxx/arle/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a>. Numbers come out of <code>scripts/bench_guidellm.sh</code> and the canonical step-driver smokes — nothing is curated.',
      rows: BENCH_ROWS_EN,
    },
    matrix: {
      title: "Support matrix",
      caption:
        'Three backends, one runtime contract. Authoritative truth lives in <a href="https://github.com/cklxx/arle/blob/main/docs/support-matrix.md"><code>docs/support-matrix.md</code></a>.',
      head: ["backend", "stability", "os / hardware", "models", "quants", "api"],
      rows: MATRIX_ROWS_EN,
    },
    files: {
      title: "Files",
      caption:
        'The repo at a glance. Everything links back to canonical paths in <code>cklxx/arle</code>.',
      rows: FILES_EN,
    },
  },
  footer: {
    left: "arle(1) · April 2026 · v0.4.2",
    right: { label: "github.com/cklxx/arle", href: "https://github.com/cklxx/arle" },
  },
};

export const ZH: Locale = {
  lang: "zh-Hans",
  hreflang: "zh-Hans",
  meta: {
    title: "arle(1) — 以 runtime 为主干的 rust workspace",
    description:
      "ARLE 是以 runtime 为主干的 Rust workspace，覆盖 CUDA、Metal、CPU 上 Qwen3 / Qwen3.5 的 serving。infer 提供 OpenAI 兼容服务；arle 是 run / serve / train / data 的统一前门。",
    ogTitle: "arle — runtime-first Rust workspace",
    ogDescription:
      "infer 在 CUDA、Metal、CPU 上提供 OpenAI 兼容 serving；arle 是 run / serve / train / data 的统一前门。",
    ogUrl: "https://cklxx.github.io/arle/zh-cn/",
    canonical: "https://cklxx.github.io/arle/zh-cn/",
  },
  masthead: {
    left: "arle(1)",
    links: [
      { label: "安装", href: "#install" },
      { label: "基准", href: "#bench" },
      { label: "矩阵", href: "#matrix" },
      { label: "github ↗", href: "https://github.com/cklxx/arle" },
    ],
  },
  hero: {
    wordmark: "arle",
    lede:
      "以 runtime 为主干的 Rust workspace。<b>infer</b> 在 CUDA、Metal、CPU 上提供 OpenAI 兼容服务；<b>arle</b> 是 run / serve / train / data 的统一前门。",
    signals: SIGNALS,
    primaryCta: { label: "$ Quickstart", href: "#install" },
    secondaryCta: { label: "cklxx/arle ↗", href: "https://github.com/cklxx/arle" },
    terminal: {
      title: "arle — bash",
      cwd: "~/projects/arle",
      lines: TERMINAL_LINES_EN,
    },
  },
  sections: {
    install: {
      title: "安装",
      caption:
        '每个平台一行能跑的命令。预编译 tarball 与 SHA 见每次 <a href="https://github.com/cklxx/arle/releases">GitHub Release</a>；curl 安装脚本会先校验 SHA256 再解压。',
      cards: INSTALL_CARDS_ZH,
    },
    bench: {
      title: "基准",
      caption:
        '直接来自 <a href="https://github.com/cklxx/arle/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> 的带日期快照。数字出自 <code>scripts/bench_guidellm.sh</code> 与标准 step-driver 冒烟，未做挑选。',
      rows: BENCH_ROWS_ZH,
    },
    matrix: {
      title: "支持矩阵",
      caption:
        '三种后端，一份运行时契约。权威矩阵见 <a href="https://github.com/cklxx/arle/blob/main/docs/support-matrix.md"><code>docs/support-matrix.md</code></a>。',
      head: ["后端", "稳定度", "系统 / 硬件", "模型", "量化", "API"],
      rows: MATRIX_ROWS_ZH,
    },
    files: {
      title: "文件",
      caption:
        '仓库一览。每条都指回 <code>cklxx/arle</code> 的标准路径。',
      rows: FILES_ZH,
    },
  },
  footer: {
    left: "arle(1) · 2026 年 4 月 · v0.4.2",
    right: { label: "github.com/cklxx/arle", href: "https://github.com/cklxx/arle" },
  },
};
