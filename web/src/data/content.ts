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
  /** Lines of the <pre> body. Raw HTML allowed (use <span class="p|c|ok|warn|k|out|dim|caret">). */
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

export type WhyCell = {
  /** Coral kicker above the heading (e.g. "conviction · 01"). */
  no: string;
  title: string;
  /** Body paragraph. Raw HTML allowed. */
  body: string;
};

export type Why = {
  caption: string; // raw HTML allowed
  cells: WhyCell[];
};

export type ArchChip = {
  label: string;
  /** bin = inverted ink, door = solid ink border, feat = dashed (feature-gated). */
  kind?: "bin" | "door" | "feat";
};

export type ArchRow = {
  /** Left layer label (e.g. "front door"). */
  layer: string;
  chips: ArchChip[];
  /** Right-hand note. Raw HTML allowed. */
  note: string;
};

export type Architecture = {
  caption: string; // raw HTML allowed
  rows: ArchRow[];
  /** Footer spans under the diagram. Raw HTML allowed. */
  foot: string[];
};

export type BattleRow = {
  /** Priority label (e.g. "P1 · active"). */
  pri: string;
  /** Coral-hot highlight on the priority label. */
  hot?: boolean;
  title: string;
  desc: string;
  /** Landing spot — crates / issue numbers. */
  where: string;
};

export type Contribute = {
  caption: string; // raw HTML allowed
  rows: BattleRow[];
  starAsk: {
    /** Lead paragraph of the dark star-ask band. Raw HTML allowed. */
    html: string;
    cta: CtaLink;
  };
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
    /** Bordered language-switch link rendered first in the nav. */
    lang: TopNavLink;
    links: TopNavLink[];
  };
  hero: {
    /** Uppercase kicker line above the lockup. */
    kicker: string;
    /** Manifesto headline. Raw HTML — use <span class="magic"> + <span class="quiet">. */
    headline: string;
    /** Lede paragraph. Raw HTML allowed. */
    lede: string;
    signals: Signal[];
    /** First entry renders as the primary (inverted) CTA. */
    ctas: CtaLink[];
    terminal: Terminal;
  };
  sections: {
    why: { title: string } & Why;
    architecture: { title: string } & Architecture;
    install: { title: string } & Install;
    bench: { title: string } & Bench;
    matrix: { title: string } & Matrix;
    contribute: { title: string } & Contribute;
    files: { title: string } & Files;
  };
  footer: {
    /** Plain-text left meta (e.g. "arle(1) · April 2026 · v0.1.5"). */
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
  { html: '<b>spec</b> dspark · mtp' },
  { html: '<b>release</b> v0.5.0 · 2026-07-26' },
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
  '<span class="dim">listening on</span> http://0.0.0.0:8000  <span class="dim">·</span> <span class="ok">ready</span> <span class="dim">in 1.4s</span><span class="caret" aria-hidden="true"></span>',
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
      '<span class="p">$</span> curl -fsSL https://github.com/acupof-ai/arle/releases/latest/download/install.sh \\',
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
      '    ghcr.io/acupof-ai/arle:latest \\',
      '    serve --backend cuda --model-path /model',
    ],
  },
  {
    label: "Source · Cargo",
    channel: "workspace",
    lines: [
      '<span class="p">$</span> git clone https://github.com/acupof-ai/arle &amp;&amp; cd arle',
      '<span class="p">$</span> cargo build --release --features cuda --bin arle',
      '<span class="c"># cli is default-on; cpu: --no-default-features --features cpu,no-cuda</span>',
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
      '<span class="p">$</span> curl -fsSL https://github.com/acupof-ai/arle/releases/latest/download/install.sh \\',
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
      '    ghcr.io/acupof-ai/arle:latest \\',
      '    serve --backend cuda --model-path /model',
    ],
  },
  {
    label: "源码 · Cargo",
    channel: "workspace",
    lines: [
      '<span class="p">$</span> git clone https://github.com/acupof-ai/arle &amp;&amp; cd arle',
      '<span class="p">$</span> cargo build --release --features cuda --bin arle',
      '<span class="c"># cli 默认开启; cpu: --no-default-features --features cpu,no-cuda</span>',
    ],
  },
];

const BENCH_ROWS_EN: BenchRow[] = [
  {
    date: "2026-07-29",
    stability: "beta",
    stabilityLabel: "beta · default-on",
    caption:
      '<b>cuda</b> · 1×H20 · <code>ThinkingCap-Qwen3.6-27B-FP8 + 27B-DFlash</code> · DSpark block 6, 48 req/point · spec decode was a c=1-only feature; the draft, rollback and capture now run batched',
    cells: [
      { key: "decode · c=1", value: "97.3", unit: "tok/s · 2.81×" },
      { key: "tok/s · c=2", value: "+88", unit: "% vs no-spec" },
      { key: "tok/s · c=16", value: "+17", unit: "% vs no-spec" },
      { key: "needle gate", value: "0", unit: "errors · exact=3 DET" },
    ],
    cmd: "arle serve --spec-type dspark · bench-agent-32k-16x8 c-sweep",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/experience/wins/2026-07-29-dspark-batched-linear-capture.md",
  },
  {
    date: "2026-07-28",
    stability: "beta",
    stabilityLabel: "beta · gate-licensed",
    caption:
      '<b>cuda</b> · 1×H20 · <code>Qwen3.6-35B-A3B-FP8</code> · 48 req/point · paged full attention launches once per layer, not once per row',
    cells: [
      { key: "ITL p50 · c=16", value: "59.5", unit: "ms · was 94.6" },
      { key: "TPOT · c=16", value: "71.2", unit: "ms · was 105.1" },
      { key: "FA3 launches / step", value: "10.0", unit: "was 157" },
      { key: "c=1", value: "±0", unit: "% · wash by design" },
    ],
    cmd: "arle serve · bench-agent-32k-16x8, matched A/B",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/experience/wins/2026-07-28-fa3-one-launch-per-layer.md",
  },
  {
    date: "2026-06-10",
    stability: "beta",
    stabilityLabel: "beta · gate-licensed",
    caption:
      '<b>cuda</b> · 8×H20 TP=8 / EP=8 · <code>DeepSeek-V4-Flash FP8</code> · B=1, official FlashMLA / DSA / DeepGEMM + MTP · 256K boots, needle-exact @230K',
    cells: [
      { key: "prefill", value: "23", unit: "ms" },
      { key: "decode", value: "15", unit: "ms/tok" },
      { key: "decode +MTP", value: "64.2", unit: "tok/s" },
      { key: "needle-exact", value: "230K", unit: "ctx" },
    ],
    cmd: "scripts/lever_gate.sh",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/experience/wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md",
  },
  {
    date: "2026-05-21",
    stability: "beta",
    stabilityLabel: "beta · cycle-wrap",
    caption: '<b>train</b> · RTX 4070 Ti SUPER 16GB · <code>Qwen3-0.6B real-checkpoint OPD step</code> · 32-commit session, kill-or-license-gated',
    cells: [
      { key: "step", value: "0.164", unit: "s" },
      { key: "vs naive CPU", value: "~170", unit: "×" },
      { key: "moderate vs PyTorch CUDA", value: "1.71", unit: "× ARLE faster" },
      { key: "held-out overlap 5k", value: "82.8", unit: "% (from 50)" },
    ],
    cmd: "cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 5000",
    href: "https://github.com/acupof-ai/arle/blob/main/CHANGELOG.md",
  },
  {
    date: "2026-05-18",
    stability: "beta",
    stabilityLabel: "beta · ad-hoc",
    caption: '<b>metal</b> · Apple M4 Pro 48GB · <code>Qwen3.6-35B-A3B 4-bit MLX</code> · HTTP serve, streaming /v1/completions',
    cells: [
      { key: "decode", value: "85.6", unit: "tok/s" },
      { key: "e2e", value: "76.1", unit: "tok/s" },
      { key: "ttft", value: "385", unit: "ms" },
      { key: "vs mlx-lm", value: "≈100", unit: "%" },
    ],
    cmd: "arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit --port 8010",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md#3-model-family-matrix",
  },
  {
    date: "2026-04-28",
    stability: "stable",
    stabilityLabel: "stable · ci-gated",
    caption: '<b>cuda</b> · NVIDIA L4 · <code>Qwen3-4B</code> · BF16 + FP8 paged KV (auto) · c=16',
    cells: [
      { key: "output", value: "197", unit: "tok/s" },
      { key: "itl p50", value: "77.9", unit: "ms" },
      { key: "vs legacy", value: "+64", unit: "%" },
      { key: "kv util", value: "69", unit: "%" },
    ],
    cmd: "python3 scripts/bench_throughput.py --concurrency-grid 16 --max-tokens 256",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md",
  },
];

const BENCH_ROWS_ZH: BenchRow[] = [
  {
    date: "2026-07-29",
    stability: "beta",
    stabilityLabel: "beta · 默认开启",
    caption:
      '<b>cuda</b> · 1×H20 · <code>ThinkingCap-Qwen3.6-27B-FP8 + 27B-DFlash</code> · DSpark block 6，48 req/point · 投机解码曾只允许 c=1；draft、rollback、capture 现在全部批处理',
    cells: [
      { key: "解码 · c=1", value: "97.3", unit: "tok/s · 2.81×" },
      { key: "tok/s · c=2", value: "+88", unit: "% 对比无投机" },
      { key: "tok/s · c=16", value: "+17", unit: "% 对比无投机" },
      { key: "needle 闸门", value: "0", unit: "错误 · exact=3 DET" },
    ],
    cmd: "arle serve --spec-type dspark · bench-agent-32k-16x8 c-sweep",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/experience/wins/2026-07-29-dspark-batched-linear-capture.md",
  },
  {
    date: "2026-07-28",
    stability: "beta",
    stabilityLabel: "beta · 闸门已 license",
    caption:
      '<b>cuda</b> · 1×H20 · <code>Qwen3.6-35B-A3B-FP8</code> · 48 req/point · 分页全注意力每层发射一次，不再每行一次',
    cells: [
      { key: "ITL p50 · c=16", value: "59.5", unit: "ms · 原 94.6" },
      { key: "TPOT · c=16", value: "71.2", unit: "ms · 原 105.1" },
      { key: "FA3 发射 / step", value: "10.0", unit: "原 157" },
      { key: "c=1", value: "±0", unit: "% · 设计上不变" },
    ],
    cmd: "arle serve · bench-agent-32k-16x8，matched A/B",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/experience/wins/2026-07-28-fa3-one-launch-per-layer.md",
  },
  {
    date: "2026-06-10",
    stability: "beta",
    stabilityLabel: "beta · 闸门已 license",
    caption:
      '<b>cuda</b> · 8×H20 TP=8 / EP=8 · <code>DeepSeek-V4-Flash FP8</code> · B=1，官方 FlashMLA / DSA / DeepGEMM + MTP · 256K 可启动、needle 230K 精确命中',
    cells: [
      { key: "prefill", value: "23", unit: "ms" },
      { key: "解码", value: "15", unit: "ms/tok" },
      { key: "解码 +MTP", value: "64.2", unit: "tok/s" },
      { key: "needle 精确", value: "230K", unit: "ctx" },
    ],
    cmd: "scripts/lever_gate.sh",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/experience/wins/2026-06-08-dsv4-decode-6ms-FINAL-consolidated.md",
  },
  {
    date: "2026-05-21",
    stability: "beta",
    stabilityLabel: "beta · 周期闭环",
    caption: '<b>train</b> · RTX 4070 Ti SUPER 16GB · <code>Qwen3-0.6B 真实 checkpoint OPD step</code> · 32 commit 单 session、kill-or-license 闸门',
    cells: [
      { key: "step", value: "0.164", unit: "s" },
      { key: "对比 naive CPU", value: "~170", unit: "×" },
      { key: "moderate vs PyTorch CUDA", value: "1.71", unit: "× ARLE 领先" },
      { key: "held-out 5k overlap", value: "82.8", unit: "%（自 50）" },
    ],
    cmd: "cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 5000",
    href: "https://github.com/acupof-ai/arle/blob/main/CHANGELOG.md",
  },
  {
    date: "2026-05-18",
    stability: "beta",
    stabilityLabel: "beta · 即席",
    caption: '<b>metal</b> · Apple M4 Pro 48GB · <code>Qwen3.6-35B-A3B 4-bit MLX</code> · HTTP serve、流式 /v1/completions',
    cells: [
      { key: "解码", value: "85.6", unit: "tok/s" },
      { key: "e2e", value: "76.1", unit: "tok/s" },
      { key: "TTFT", value: "385", unit: "ms" },
      { key: "对比 mlx-lm", value: "≈100", unit: "%" },
    ],
    cmd: "arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit --port 8010",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md#3-model-family-matrix",
  },
  {
    date: "2026-04-28",
    stability: "stable",
    stabilityLabel: "stable · CI 已门控",
    caption: '<b>cuda</b> · NVIDIA L4 · <code>Qwen3-4B</code> · BF16 + FP8 分页 KV（auto）· c=16',
    cells: [
      { key: "输出", value: "197", unit: "tok/s" },
      { key: "ITL p50", value: "77.9", unit: "ms" },
      { key: "对比 legacy", value: "+64", unit: "%" },
      { key: "KV 利用率", value: "69", unit: "%" },
    ],
    cmd: "python3 scripts/bench_throughput.py --concurrency-grid 16 --max-tokens 256",
    href: "https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md",
  },
];

const MATRIX_ROWS_EN: string[][] = [
  [
    "<code>cuda</code>",
    '<span class="pill ok">stable</span>',
    "Linux + NVIDIA Ampere+",
    "Qwen3 / Qwen3.5 / Qwen3.6 · DeepSeek-V4-Flash (8×H20)",
    "FP16 / BF16 · FP8 KV (auto) · GGUF Q4_K",
    "OpenAI v1",
  ],
  [
    "<code>metal</code>",
    '<span class="pill warn">beta</span>',
    "Apple Silicon (M1+)",
    "Qwen3.5 · Qwen3.6 MoE (canonical)",
    "FP16 / BF16 · MLX 4-bit · GGUF Q4_K",
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
    "Qwen3 / Qwen3.5 / Qwen3.6 · DeepSeek-V4-Flash（8×H20）",
    "FP16 / BF16 · FP8 KV（auto）· GGUF Q4_K",
    "OpenAI v1",
  ],
  [
    "<code>metal</code>",
    '<span class="pill warn">beta</span>',
    "Apple Silicon（M1+）",
    "Qwen3.5 · Qwen3.6 MoE（canonical）",
    "FP16 / BF16 · MLX 4-bit · GGUF Q4_K",
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

const WHY_CELLS_EN: WhyCell[] = [
  {
    no: "conviction · 01",
    title: "From scratch is the point.",
    body:
      "From-scratch autograd, scheduler, radix prefix cache, paged KV, CUDA graphs, MLX bridge. The concepts you’ve only met in papers are here as small Rust files — <b>read one crate, own one concept</b>.",
  },
  {
    no: "conviction · 02",
    title: "Kills are documented.",
    body:
      "Dead ends stay in the repo, marked <code>KILL</code> with the measurement that killed them. You inherit the measurements, not just the conclusions — <b>months of A/B work, free to read</b>.",
  },
  {
    no: "conviction · 03",
    title: "Numbers are dated.",
    body:
      'Every benchmark is a dated snapshot in <a href="https://github.com/acupof-ai/arle/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> with env, params, and regressions. Nothing is curated. “Fast” is a number with a date, or it’s nothing.',
  },
  {
    no: "conviction · 04",
    title: "One binary, no glue.",
    body:
      "No Python at serving time, no sidecar processes, no config sprawl. <code>arle</code> is the only binary the workspace builds — <b>clone to first token in minutes</b>, on a GPU box or a MacBook.",
  },
];

const WHY_CELLS_ZH: WhyCell[] = [
  {
    no: "conviction · 01",
    title: "从零写起，就是目的。",
    body:
      "自研 autograd、调度器、radix 前缀缓存、paged KV、CUDA graphs、MLX bridge。论文里见过的概念，在这里是一个个小 Rust 文件 —— <b>读一个 crate，吃透一个概念</b>。",
  },
  {
    no: "conviction · 02",
    title: "被砍掉的也留档。",
    body:
      "死路留在仓库里，标着 <code>KILL</code> 和杀死它的测量数据。你继承的是测量过程，不只是结论 —— <b>几个月的 A/B 实验，免费可读</b>。",
  },
  {
    no: "conviction · 03",
    title: "数字都有日期。",
    body:
      '每个基准都是 <a href="https://github.com/acupof-ai/arle/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> 里带日期的快照，环境、参数、回归俱全。没有精选。「快」要么是带日期的数字，要么什么都不是。',
  },
  {
    no: "conviction · 04",
    title: "一个二进制，零胶水。",
    body:
      "serving 时没有 Python，没有 sidecar 进程，没有配置蔓延。<code>arle</code> 是 workspace 唯一构建的二进制 —— <b>从 clone 到第一个 token 只要几分钟</b>，GPU 机器或 MacBook 都行。",
  },
];

const ARCH_ROWS: ArchRow[] = [
  {
    layer: "bin",
    chips: [{ label: "arle", kind: "bin" }],
    note: "the only binary the workspace builds · <code>src/main.rs</code>",
  },
  {
    layer: "control plane",
    chips: [{ label: "cli" }, { label: "agent" }, { label: "chat" }, { label: "tools" }],
    note: "REPL · session loop · protocol · sandboxed tools",
  },
  {
    layer: "front door",
    chips: [{ label: "infer-api", kind: "door" }],
    note: "<code>InferenceEngine</code> · <code>LoadedInferenceEngine</code> · backends plug in here",
  },
  {
    layer: "server · core",
    chips: [{ label: "infer-server" }, { label: "infer-core" }],
    note: "OpenAI v1 facade (axum) · <code>Engine&lt;E,K&gt;</code> · scheduler · radix prefix",
  },
  {
    layer: "seam · ir",
    chips: [{ label: "infer-seam" }, { label: "infer-plan" }],
    note: "<code>BackendExecutor</code> + <code>KvPool</code> seam · <code>ForwardPlan</code> IR — host-only",
  },
  {
    layer: "backends",
    chips: [{ label: "infer-cuda", kind: "feat" }, { label: "infer-metal", kind: "feat" }],
    note: "feature-gated · metal’s host KV pool doubles as the cpu smoke path",
  },
  {
    layer: "kernels",
    chips: [{ label: "cuda-kernels" }, { label: "mlx-sys" }, { label: "kv-native-sys" }],
    note: "CUDA C / TileLang · MLX C++ bridge · KV persistence",
  },
];

const ARCH_FOOT_EN: string[] = [
  "<b>pure leaves</b> · infer-topo · infer-moe · infer-util",
  "<b>specs</b> · qwen3 · qwen35 · deepseek",
  "<b>ffi</b> · deepep-sys · xgrammar-sys",
  "<b>train</b> · autograd + train — OPD-only since 2026-05-18",
];

const ARCH_FOOT_ZH: string[] = [
  "<b>纯叶子</b> · infer-topo · infer-moe · infer-util",
  "<b>specs</b> · qwen3 · qwen35 · deepseek",
  "<b>ffi</b> · deepep-sys · xgrammar-sys",
  "<b>train</b> · autograd + train — 2026-05-18 起仅 OPD",
];

const BATTLE_ROWS_EN: BattleRow[] = [
  {
    pri: "P1 · active",
    hot: true,
    title: "256K OPD on one GPU",
    desc: "seq-chunked recompute shipped behind one exact fold; the verdict wants completed 65536/131072 steps, and the remaining O(seq) wall is the 48 un-chunked linear-attention layers",
    where: "autograd · train · #188 #189",
  },
  {
    pri: "P2 · active",
    title: "DSpark goodput scheduler",
    desc: "Algorithm-1 verify budget shipped 2026-07-30; open walls — the verify path's per-step D2H + sync (645 MB/slot scratch against a 6-row reality) and a slot budget that starves KV",
    where: "infer-cuda · #182 #183 #184",
  },
  {
    pri: "P3 · queued",
    title: "DSpark on DSv4, 8×H20",
    desc: "C5 end-to-end DSpark serve on the DSv4 chain — license-or-kill against the shipped MTP draft on the same agent fingerprint",
    where: "infer-cuda · docs/plans · #128 #123",
  },
  {
    pri: "open",
    title: "Train through the inference operators",
    desc: "Path B routes the training forward through the serving operator set — kills the duplicated-kernel SIGFPE class and unlocks a 35B MoE student",
    where: "autograd · train · #103",
  },
  {
    pri: "open",
    title: "Third backends: HIP / Vulkan",
    desc: "HIP substrate and a coherent Vulkan forward (gfx1151, 3.0 s/tok) landed; the license is perf parity, not a boot",
    where: "infer-hip · infer-vulkan · #71",
  },
];

const BATTLE_ROWS_ZH: BattleRow[] = [
  {
    pri: "P1 · active",
    hot: true,
    title: "单卡 256K OPD",
    desc: "seq-chunked recompute 已随精确 fold 落地；verdict 还差 65536/131072 的完整 step，剩余的 O(seq) 墙是 48 层未 chunk 的线性注意力",
    where: "autograd · train · #188 #189",
  },
  {
    pri: "P2 · active",
    title: "DSpark goodput 调度器",
    desc: "Algorithm-1 verify budget 2026-07-30 落地；下一批墙：verify 路径逐步的 D2H + sync（6 行实际需求对着 645 MB/slot 的 scratch）、以及饿死 KV 的 slot 预算",
    where: "infer-cuda · #182 #183 #184",
  },
  {
    pri: "P3 · queued",
    title: "DSv4 上跑 DSpark（8×H20）",
    desc: "C5 —— DSv4 链路上端到端 DSpark serve，同一份 agent 指纹对 shipped MTP draft 做 license-or-kill",
    where: "infer-cuda · docs/plans · #128 #123",
  },
  {
    pri: "open",
    title: "训练前向走推理算子",
    desc: "Path B 把训练前向路由到 serving 算子集 —— 消灭重复 kernel 的 SIGFPE 一类问题，解锁 35B MoE student",
    where: "autograd · train · #103",
  },
  {
    pri: "open",
    title: "第三后端：HIP / Vulkan",
    desc: "HIP 基板与 coherent 的 Vulkan 前向（gfx1151，3.0 s/tok）已落地；license 条件是性能 parity，不是能跑",
    where: "infer-hip · infer-vulkan · #71",
  },
];

const FILES_EN: FileRow[] = [
  { path: "/README.md", desc: "public overview · install · CLI · architecture", href: "https://github.com/acupof-ai/arle/blob/main/README.md" },
  { path: "/docs/codebase-map.md", desc: "canonical workspace topology", href: "https://github.com/acupof-ai/arle/blob/main/docs/codebase-map.md" },
  { path: "/docs/http-api.md", desc: "HTTP contract · streaming behavior", href: "https://github.com/acupof-ai/arle/blob/main/docs/http-api.md" },
  { path: "/docs/support-matrix.md", desc: "backend / model / quant support", href: "https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md" },
  { path: "/docs/experience/wins/", desc: "dated benchmark snapshots", href: "https://github.com/acupof-ai/arle/tree/main/docs/experience/wins" },
  { path: "/crates/cli/", desc: "arle binary · verbs · doctor", href: "https://github.com/acupof-ai/arle/tree/main/crates/cli" },
  { path: "/crates/infer-api/", desc: "the single public front door", href: "https://github.com/acupof-ai/arle/tree/main/crates/infer-api" },
  { path: "/crates/infer-core/", desc: "runtime spine · engine · scheduler · radix cache", href: "https://github.com/acupof-ai/arle/tree/main/crates/infer-core" },
  { path: "/crates/train/", desc: "OPD loop · autograd tape · seq-chunked recompute", href: "https://github.com/acupof-ai/arle/tree/main/crates/train" },
  { path: "/crates/cuda-kernels/", desc: "cuda kernel crate · csrc · prelude", href: "https://github.com/acupof-ai/arle/tree/main/crates/cuda-kernels" },
  { path: "/crates/mlx-sys/", desc: "metal bridge · cmake + cc", href: "https://github.com/acupof-ai/arle/tree/main/crates/mlx-sys" },
  { path: "/examples/", desc: "copyable curl · Docker · Metal · tiny train smokes", href: "https://github.com/acupof-ai/arle/tree/main/examples" },
  { path: "/releases", desc: "tagged binaries · checksums", href: "https://github.com/acupof-ai/arle/releases" },
];

const FILES_ZH: FileRow[] = [
  { path: "/README.zh-CN.md", desc: "中文公共入口：安装 · CLI · 架构", href: "https://github.com/acupof-ai/arle/blob/main/README.zh-CN.md" },
  { path: "/docs/codebase-map.md", desc: "权威 workspace 拓扑", href: "https://github.com/acupof-ai/arle/blob/main/docs/codebase-map.md" },
  { path: "/docs/http-api.md", desc: "HTTP 契约 · 流式行为", href: "https://github.com/acupof-ai/arle/blob/main/docs/http-api.md" },
  { path: "/docs/support-matrix.md", desc: "后端 / 模型 / 量化支持", href: "https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md" },
  { path: "/docs/experience/wins/", desc: "带日期的基准快照", href: "https://github.com/acupof-ai/arle/tree/main/docs/experience/wins" },
  { path: "/crates/cli/", desc: "arle 二进制 · 子命令 · doctor", href: "https://github.com/acupof-ai/arle/tree/main/crates/cli" },
  { path: "/crates/infer-api/", desc: "唯一的公共前门", href: "https://github.com/acupof-ai/arle/tree/main/crates/infer-api" },
  { path: "/crates/infer-core/", desc: "运行时主干 · engine · scheduler · radix cache", href: "https://github.com/acupof-ai/arle/tree/main/crates/infer-core" },
  { path: "/crates/train/", desc: "OPD 循环 · autograd tape · seq-chunked recompute", href: "https://github.com/acupof-ai/arle/tree/main/crates/train" },
  { path: "/crates/cuda-kernels/", desc: "cuda kernel crate · csrc · prelude", href: "https://github.com/acupof-ai/arle/tree/main/crates/cuda-kernels" },
  { path: "/crates/mlx-sys/", desc: "metal 桥接 · cmake + cc", href: "https://github.com/acupof-ai/arle/tree/main/crates/mlx-sys" },
  { path: "/examples/", desc: "curl · Docker · Metal · tiny train 冒烟示例", href: "https://github.com/acupof-ai/arle/tree/main/examples" },
  { path: "/releases", desc: "发版二进制 · 校验和", href: "https://github.com/acupof-ai/arle/releases" },
];

export const EN: Locale = {
  lang: "en",
  hreflang: "en",
  meta: {
    title: "arle(1) — runtime-first rust workspace",
    description:
      "ARLE is a runtime-first Rust workspace for serving Qwen3.5 / Qwen3.6 and DeepSeek-V4-Flash on CUDA, Metal, and CPU. The whole modern inference stack — continuous batching, radix prefix cache, paged KV, CUDA graphs, speculative decode — hand-built in Rust.",
    ogTitle: "arle — inference stops being magic",
    ogDescription:
      "The whole modern inference stack — continuous batching, radix prefix cache, paged KV, CUDA graphs, speculative decode — hand-built in Rust, small enough to read in a weekend.",
    ogUrl: "https://acupof-ai.github.io/arle/",
    canonical: "https://acupof-ai.github.io/arle/",
  },
  masthead: {
    left: "arle(1)",
    lang: { label: "中文", href: "/arle/zh-cn/" },
    links: [
      { label: "why", href: "#why" },
      { label: "architecture", href: "#architecture" },
      { label: "install", href: "#install" },
      { label: "bench", href: "#bench" },
      { label: "contribute", href: "#contribute" },
      { label: "github ↗", href: "https://github.com/acupof-ai/arle" },
    ],
  },
  hero: {
    kicker: "one person · pure rust · from scratch · in public",
    headline:
      'Inference stops<br>being <span class="magic">magic</span><span class="quiet">.</span>',
    lede:
      "The whole modern inference stack — continuous batching, radix prefix cache, paged KV, CUDA graphs, speculative decode — hand-built in Rust, <b>small enough to read in a weekend</b>.",
    signals: SIGNALS,
    ctas: [
      { label: "★ Star acupof-ai/arle", href: "https://github.com/acupof-ai/arle" },
      { label: "$ Quickstart", href: "#install" },
      { label: "contribute →", href: "#contribute" },
    ],
    terminal: {
      title: "arle — bash",
      cwd: "~/projects/arle",
      lines: TERMINAL_LINES_EN,
    },
  },
  sections: {
    why: {
      title: "Why this exists",
      caption:
        "Not a wrapper, not a binding, not a fork. Every layer the big engines hide behind a pip install exists here as <b>a few hundred lines you can step through</b> — and a readable trail of why each one looks the way it does.",
      cells: WHY_CELLS_EN,
    },
    architecture: {
      title: "Architecture",
      caption:
        'Post-cutover (2026-06-04) the monolithic <code>infer</code> crate is gone. The runtime is a device-neutral crate graph — dependencies flow strictly downward, <b>infer-core carries no backend dependency</b>, and backends plug in at the front door. Canonical topology lives in <a href="https://github.com/acupof-ai/arle/blob/main/docs/codebase-map.md"><code>docs/codebase-map.md</code></a>.',
      rows: ARCH_ROWS,
      foot: ARCH_FOOT_EN,
    },
    install: {
      title: "Install",
      caption:
        'One runnable line per platform. Pre-built tarballs and SHAs on each <a href="https://github.com/acupof-ai/arle/releases">GitHub Release</a>; the curl installer verifies SHA256 before extracting.',
      cards: INSTALL_CARDS_EN,
    },
    bench: {
      title: "Bench",
      caption:
        'Dated, reproducible snapshots straight from <a href="https://github.com/acupof-ai/arle/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a>. Numbers come out of <code>scripts/bench_throughput.py</code> and the <code>bench-agent</code> workload driver — nothing is curated.',
      rows: BENCH_ROWS_EN,
    },
    matrix: {
      title: "Support matrix",
      caption:
        'Three backends, one runtime contract. Authoritative truth lives in <a href="https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md"><code>docs/support-matrix.md</code></a>.',
      head: ["backend", "stability", "os / hardware", "models", "quants", "api"],
      rows: MATRIX_ROWS_EN,
    },
    contribute: {
      title: "Where a contribution lands",
      caption:
        'No queue, no committee — <b>a weekend PR here can move a headline number</b>, and the battlefields are public: one tracked <a href="https://github.com/acupof-ai/arle/issues">GitHub issue</a> per front. Start with <a href="https://github.com/acupof-ai/arle/blob/main/CONTRIBUTING.md"><code>CONTRIBUTING.md</code></a>.',
      rows: BATTLE_ROWS_EN,
      starAsk: {
        html:
          "<b>Stars are the only metric a solo project has.</b> If this repo saved you a read of someone else’s CUDA — or just proved it can be done in Rust — leave one. It decides how much time this gets.",
        cta: { label: "★ Star acupof-ai/arle", href: "https://github.com/acupof-ai/arle" },
      },
    },
    files: {
      title: "Files",
      caption:
        'The repo at a glance. Everything links back to canonical paths in <code>acupof-ai/arle</code>.',
      rows: FILES_EN,
    },
  },
  footer: {
    left: "arle(1) · July 2026 · v0.5.0",
    right: { label: "github.com/acupof-ai/arle", href: "https://github.com/acupof-ai/arle" },
  },
};

export const ZH: Locale = {
  lang: "zh-Hans",
  hreflang: "zh-Hans",
  meta: {
    title: "arle(1) — 以 runtime 为主干的 rust workspace",
    description:
      "ARLE 是以 runtime 为主干的 Rust workspace，覆盖 CUDA、Metal、CPU 上 Qwen3.5 / Qwen3.6 与 DeepSeek-V4-Flash 的 serving。现代推理栈的全部 —— continuous batching、radix 前缀缓存、paged KV、CUDA graphs、投机解码 —— 一个人用 Rust 手写。",
    ogTitle: "arle — 推理，不再是魔法",
    ogDescription:
      "现代推理栈的全部 —— continuous batching、radix 前缀缓存、paged KV、CUDA graphs、投机解码 —— 一个人用 Rust 手写，小到一个周末就能读完。",
    ogUrl: "https://acupof-ai.github.io/arle/zh-cn/",
    canonical: "https://acupof-ai.github.io/arle/zh-cn/",
  },
  masthead: {
    left: "arle(1)",
    lang: { label: "EN", href: "/arle/" },
    links: [
      { label: "理念", href: "#why" },
      { label: "架构", href: "#architecture" },
      { label: "安装", href: "#install" },
      { label: "基准", href: "#bench" },
      { label: "参与", href: "#contribute" },
      { label: "github ↗", href: "https://github.com/acupof-ai/arle" },
    ],
  },
  hero: {
    kicker: "一个人 · 纯 rust · 从零写起 · 公开构建",
    headline:
      '推理，不再是<br><span class="magic">魔法</span><span class="quiet">。</span>',
    lede:
      "现代推理栈的全部 —— continuous batching、radix 前缀缓存、paged KV、CUDA graphs、投机解码 —— 一个人用 Rust 手写，<b>小到一个周末就能读完</b>。",
    signals: SIGNALS,
    ctas: [
      { label: "★ Star acupof-ai/arle", href: "https://github.com/acupof-ai/arle" },
      { label: "$ Quickstart", href: "#install" },
      { label: "参与贡献 →", href: "#contribute" },
    ],
    terminal: {
      title: "arle — bash",
      cwd: "~/projects/arle",
      lines: TERMINAL_LINES_EN,
    },
  },
  sections: {
    why: {
      title: "为什么做这个",
      caption:
        "不是 wrapper，不是 binding，不是 fork。大引擎藏在 pip install 背后的每一层，在这里都是<b>几百行可以单步调试的 Rust</b> —— 还有每个决定为什么长这样的完整记录。",
      cells: WHY_CELLS_ZH,
    },
    architecture: {
      title: "架构",
      caption:
        '2026-06-04 cutover 之后，单体 <code>infer</code> crate 已经不在了。运行时是一张设备无关的 crate 图 —— 依赖严格向下流动，<b>infer-core 不依赖任何后端</b>，后端在前门接入。权威拓扑见 <a href="https://github.com/acupof-ai/arle/blob/main/docs/codebase-map.md"><code>docs/codebase-map.md</code></a>。',
      rows: ARCH_ROWS,
      foot: ARCH_FOOT_ZH,
    },
    install: {
      title: "安装",
      caption:
        '每个平台一行能跑的命令。预编译 tarball 与 SHA 见每次 <a href="https://github.com/acupof-ai/arle/releases">GitHub Release</a>；curl 安装脚本会先校验 SHA256 再解压。',
      cards: INSTALL_CARDS_ZH,
    },
    bench: {
      title: "基准",
      caption:
        '直接来自 <a href="https://github.com/acupof-ai/arle/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> 的带日期快照。数字出自 <code>scripts/bench_throughput.py</code> 与 <code>bench-agent</code> 工作负载驱动，未做挑选。',
      rows: BENCH_ROWS_ZH,
    },
    matrix: {
      title: "支持矩阵",
      caption:
        '三种后端，一份运行时契约。权威矩阵见 <a href="https://github.com/acupof-ai/arle/blob/main/docs/support-matrix.md"><code>docs/support-matrix.md</code></a>。',
      head: ["后端", "稳定度", "系统 / 硬件", "模型", "量化", "API"],
      rows: MATRIX_ROWS_ZH,
    },
    contribute: {
      title: "贡献会落在哪",
      caption:
        '没有排队，没有委员会 —— <b>一个周末的 PR 就能改动头条数字</b>，战场全部公开：每条战线一个 <a href="https://github.com/acupof-ai/arle/issues">GitHub 跟踪 issue</a>。从 <a href="https://github.com/acupof-ai/arle/blob/main/CONTRIBUTING.md"><code>CONTRIBUTING.md</code></a> 开始。',
      rows: BATTLE_ROWS_ZH,
      starAsk: {
        html:
          "<b>Star 是个人项目唯一的指标。</b>如果这个仓库帮你省下了啃别人 CUDA 的时间 —— 或者只是证明了这件事 Rust 写得出来 —— 留一颗。它决定这件事能得到多少时间。",
        cta: { label: "★ Star acupof-ai/arle", href: "https://github.com/acupof-ai/arle" },
      },
    },
    files: {
      title: "文件",
      caption:
        '仓库一览。每条都指回 <code>acupof-ai/arle</code> 的标准路径。',
      rows: FILES_ZH,
    },
  },
  footer: {
    left: "arle(1) · 2026 年 7 月 · v0.5.0",
    right: { label: "github.com/acupof-ai/arle", href: "https://github.com/acupof-ai/arle" },
  },
};
