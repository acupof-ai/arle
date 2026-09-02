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

const GH = "https://github.com/acupof-ai/arle";
const WINS = `${GH}/blob/main/docs/experience/wins`;

const SIGNALS: Signal[] = [
  { html: '<b>api</b> anthropic /v1/messages · openai v1' },
  { html: '<b>metal</b> beta · apple silicon', dot: "warn" },
  { html: '<b>cuda</b> stable · ampere+', dot: "ok" },
  { html: '<b>kv</b> prefix cache survives turns' },
  { html: '<b>spec</b> mtp · dspark · bit-identical' },
  { html: '<b>release</b> v0.5.8 · 2026-08-21' },
];

const TERMINAL_LINES_EN = [
  '<span class="p">$</span> arle <span class="k">serve</span> --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit',
  '<span class="dim">serving on</span> http://127.0.0.1:8000  <span class="dim">·</span> /v1/messages <span class="dim">·</span> /v1/chat/completions',
  "",
  '<span class="p">$</span> ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=local claude',
  '<span class="c"># turn 1: cold prefill of the system prompt</span>',
  '<span class="out">prefix-lookup</span>  prompt=4850 licensed_blocks=<span class="warn">0</span>',
  '<span class="c"># turn 2 … turn 12: only the new tokens prefill</span>',
  '<span class="out">prefix-attach</span>  matched=<span class="ok">8640</span> restored=<span class="ok">8640</span> committed=8651<span class="caret" aria-hidden="true"></span>',
];

const INSTALL_CARDS_EN: InstallCard[] = [
  {
    label: "Apple Silicon · Homebrew",
    channel: "zsh / bash",
    lines: [
      '<span class="p">$</span> brew install cklxx/tap/arle',
      '<span class="p">$</span> arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit',
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
      '    -v $PWD/models:/models:ro ghcr.io/acupof-ai/arle:latest \\',
      '    serve --backend cuda --model-path /models/Qwen3.6-27B',
    ],
  },
  {
    label: "Connect · Claude Code / OpenAI clients",
    channel: "any shell",
    lines: [
      '<span class="p">$</span> ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=local claude',
      '<span class="p">$</span> export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=local',
      '<span class="c"># opencode, aider, the openai SDK — anything that speaks the OpenAI API</span>',
    ],
  },
];

const INSTALL_CARDS_ZH: InstallCard[] = [
  {
    label: "Apple Silicon · Homebrew",
    channel: "zsh / bash",
    lines: [
      '<span class="p">$</span> brew install cklxx/tap/arle',
      '<span class="p">$</span> arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit',
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
      '    -v $PWD/models:/models:ro ghcr.io/acupof-ai/arle:latest \\',
      '    serve --backend cuda --model-path /models/Qwen3.6-27B',
    ],
  },
  {
    label: "接入 · Claude Code / OpenAI 客户端",
    channel: "任意 shell",
    lines: [
      '<span class="p">$</span> ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=local claude',
      '<span class="p">$</span> export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=local',
      '<span class="c"># opencode、aider、openai SDK —— 任何说 OpenAI API 的工具</span>',
    ],
  },
];

const BENCH_ROWS_EN: BenchRow[] = [
  {
    date: "2026-09-02",
    stability: "beta",
    stabilityLabel: "beta · default-on",
    caption:
      '<b>metal</b> · M4 Pro 48 GB · <code>Qwen3.5-0.8B-MLX-4bit</code> · 12-turn agent conversation, 4.8K-token system prompt, +~350 tokens per turn · same weights and request bytes for both servers',
    cells: [
      { key: "TTFT · turns 2–12 median", value: "180", unit: "ms" },
      { key: "mlx-lm 0.31.2 · same", value: "249", unit: "ms" },
      { key: "TTFT · turn 12", value: "202", unit: "ms · 8.6K tokens" },
      { key: "restored vs cold", value: "18/18", unit: "needle exact · DET" },
    ],
    cmd: "scripts/bench_multiturn_ttft.py --turns 12 --warmup",
    href: `${WINS}/2026-09-02-metal-prefix-restore-survives-turns.md`,
  },
  {
    date: "2026-06-14",
    stability: "beta",
    stabilityLabel: "beta · snapshot",
    caption:
      '<b>metal</b> · M4 Pro 48 GB · <code>Qwen3.6-35B-A3B-4bit</code> (MoE, ~3B active) · 512-in / 128-out · single stream · median of 6',
    cells: [
      { key: "decode", value: "85.3", unit: "tok/s" },
      { key: "time per token", value: "11.7", unit: "ms" },
      { key: "TTFT · 512 tokens", value: "1.23", unit: "s" },
      { key: "Qwen3.5-0.8B decode", value: "318", unit: "tok/s" },
    ],
    cmd: "arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit",
    href: `${GH}/blob/main/benchmarks/README.md`,
  },
  {
    date: "2026-08-14",
    stability: "stable",
    stabilityLabel: "stable · anchor",
    caption:
      '<b>cuda</b> · 1×H20 · <code>Qwen3.6-27B-FP8</code> + block-drafter speculative decode (DSpark) · 32K-token multi-turn agent prompts · per-request decode',
    cells: [
      { key: "decode · c=1", value: "91.8", unit: "tok/s" },
      { key: "decode · c=8", value: "20.5", unit: "tok/s" },
      { key: "35B-A3B MoE · c=1", value: "149.3", unit: "tok/s" },
      { key: "vs SGLang 0.5.13 · decode", value: "−2.8", unit: "% per token" },
    ],
    cmd: "arle serve --backend cuda --spec-type dspark · bench-agent-32k",
    href: `${GH}/blob/main/docs/baselines.md`,
  },
  {
    date: "2026-08-20",
    stability: "beta",
    stabilityLabel: "beta · default-on",
    caption:
      '<b>cuda</b> · 1×H20 · <code>Qwen3.8-27B-NVFP4</code> vs <code>Qwen3.6-27B-FP8</code> · same binary, both arms back to back · nothing resident twice',
    cells: [
      { key: "decode · c=1", value: "+21.3", unit: "% vs FP8" },
      { key: "end-to-end · c=4", value: "+15.3", unit: "% vs FP8" },
      { key: "resident", value: "22.4", unit: "GB · FP8 29.4" },
      { key: "GSM8K-shaped", value: "188/200", unit: "FP8 189/200" },
    ],
    cmd: "arle serve --backend cuda --model-path unsloth/Qwen3.8-27B-NVFP4",
    href: `${WINS}/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md`,
  },
  {
    date: "2026-08-23",
    stability: "beta",
    stabilityLabel: "beta · default-on",
    caption:
      '<b>cuda</b> · 4×H20 TP=4 · <code>DeepSeek-V4-Flash</code> · c=1 decode body captured into one CUDA graph per slot · 32K agent prompts',
    cells: [
      { key: "decode · NVFP4 experts", value: "44.2", unit: "tok/s · was 40.8" },
      { key: "decode · FP8 experts", value: "59.5", unit: "tok/s · was 52.4" },
      { key: "ITL p50", value: "22.2", unit: "ms · was 24.1" },
      { key: "MMLU · 200 items", value: "0", unit: "per-item diffs" },
    ],
    cmd: "arle serve --backend cuda --tensor-parallel-size 4",
    href: `${WINS}/2026-08-23-dsv4-c1-decode-graph.md`,
  },
  {
    date: "2026-06-20",
    stability: "beta",
    stabilityLabel: "beta · multi-seed",
    caption:
      '<b>train</b> · On-Policy Distillation · the teacher is the serving engine, the student trains on its own rollouts · <code>Qwen3.5-4B</code> and <code>Qwen3.5-27B</code>',
    cells: [
      { key: "MATH-500 · 4B", value: "+27", unit: "pp · 0.518 → 0.792" },
      { key: "Terminal-Bench · 27B", value: "+5.1", unit: "pp pass@1" },
      { key: "BFCL-live abstention", value: "1.00", unit: "from 0.60" },
      { key: "python on the hot path", value: "0", unit: "processes" },
    ],
    cmd: "arle train opd",
    href: `${WINS}/2026-06-20-opd-multiseed-math500-lock.md`,
  },
];

const BENCH_ROWS_ZH: BenchRow[] = [
  {
    date: "2026-09-02",
    stability: "beta",
    stabilityLabel: "beta · 默认开启",
    caption:
      '<b>metal</b> · M4 Pro 48 GB · <code>Qwen3.5-0.8B-MLX-4bit</code> · 12 轮 agent 对话，4.8K token 系统提示，每轮追加约 350 token · 两台服务器权重与请求字节完全一致',
    cells: [
      { key: "TTFT · 第 2 到 12 轮中位数", value: "180", unit: "ms" },
      { key: "mlx-lm 0.31.2 · 同条件", value: "249", unit: "ms" },
      { key: "TTFT · 第 12 轮", value: "202", unit: "ms · 8.6K token" },
      { key: "恢复 vs 冷启", value: "18/18", unit: "needle 精确 · DET" },
    ],
    cmd: "scripts/bench_multiturn_ttft.py --turns 12 --warmup",
    href: `${WINS}/2026-09-02-metal-prefix-restore-survives-turns.md`,
  },
  {
    date: "2026-06-14",
    stability: "beta",
    stabilityLabel: "beta · 快照",
    caption:
      '<b>metal</b> · M4 Pro 48 GB · <code>Qwen3.6-35B-A3B-4bit</code>（MoE，约 3B 激活）· 512-in / 128-out · 单流 · 6 次中位数',
    cells: [
      { key: "解码", value: "85.3", unit: "tok/s" },
      { key: "每 token", value: "11.7", unit: "ms" },
      { key: "TTFT · 512 token", value: "1.23", unit: "s" },
      { key: "Qwen3.5-0.8B 解码", value: "318", unit: "tok/s" },
    ],
    cmd: "arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit",
    href: `${GH}/blob/main/benchmarks/README.md`,
  },
  {
    date: "2026-08-14",
    stability: "stable",
    stabilityLabel: "stable · 锚点",
    caption:
      '<b>cuda</b> · 1×H20 · <code>Qwen3.6-27B-FP8</code> + 块草稿推测解码（DSpark）· 32K token 多轮 agent 提示 · 单请求解码',
    cells: [
      { key: "解码 · c=1", value: "91.8", unit: "tok/s" },
      { key: "解码 · c=8", value: "20.5", unit: "tok/s" },
      { key: "35B-A3B MoE · c=1", value: "149.3", unit: "tok/s" },
      { key: "对比 SGLang 0.5.13 · 解码", value: "−2.8", unit: "% 每 token" },
    ],
    cmd: "arle serve --backend cuda --spec-type dspark · bench-agent-32k",
    href: `${GH}/blob/main/docs/baselines.md`,
  },
  {
    date: "2026-08-20",
    stability: "beta",
    stabilityLabel: "beta · 默认开启",
    caption:
      '<b>cuda</b> · 1×H20 · <code>Qwen3.8-27B-NVFP4</code> 对 <code>Qwen3.6-27B-FP8</code> · 同一二进制两条臂背靠背 · 权重不存两份',
    cells: [
      { key: "解码 · c=1", value: "+21.3", unit: "% 对 FP8" },
      { key: "端到端 · c=4", value: "+15.3", unit: "% 对 FP8" },
      { key: "常驻", value: "22.4", unit: "GB · FP8 29.4" },
      { key: "GSM8K 形状", value: "188/200", unit: "FP8 189/200" },
    ],
    cmd: "arle serve --backend cuda --model-path unsloth/Qwen3.8-27B-NVFP4",
    href: `${WINS}/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md`,
  },
  {
    date: "2026-08-23",
    stability: "beta",
    stabilityLabel: "beta · 默认开启",
    caption:
      '<b>cuda</b> · 4×H20 TP=4 · <code>DeepSeek-V4-Flash</code> · c=1 解码主体按 slot 捕获成一张 CUDA graph · 32K agent 提示',
    cells: [
      { key: "解码 · NVFP4 专家", value: "44.2", unit: "tok/s · 原 40.8" },
      { key: "解码 · FP8 专家", value: "59.5", unit: "tok/s · 原 52.4" },
      { key: "ITL p50", value: "22.2", unit: "ms · 原 24.1" },
      { key: "MMLU · 200 题", value: "0", unit: "逐题差异" },
    ],
    cmd: "arle serve --backend cuda --tensor-parallel-size 4",
    href: `${WINS}/2026-08-23-dsv4-c1-decode-graph.md`,
  },
  {
    date: "2026-06-20",
    stability: "beta",
    stabilityLabel: "beta · 多种子",
    caption:
      '<b>train</b> · On-Policy Distillation · teacher 就是 serving 引擎，student 在自己的 rollout 上训练 · <code>Qwen3.5-4B</code> 与 <code>Qwen3.5-27B</code>',
    cells: [
      { key: "MATH-500 · 4B", value: "+27", unit: "pp · 0.518 → 0.792" },
      { key: "Terminal-Bench · 27B", value: "+5.1", unit: "pp pass@1" },
      { key: "BFCL-live 弃答", value: "1.00", unit: "自 0.60" },
      { key: "热路径上的 python", value: "0", unit: "个进程" },
    ],
    cmd: "arle train opd",
    href: `${WINS}/2026-06-20-opd-multiseed-math500-lock.md`,
  },
];

const MATRIX_ROWS_EN: string[][] = [
  [
    "<code>cuda</code>",
    '<span class="pill ok">stable</span>',
    "Linux + NVIDIA Ampere+ · 1–8 GPUs (TP / EP)",
    "Qwen3.5 / 3.6 / 3.8 · DeepSeek-V4-Flash · GLM-5.2",
    "BF16 · FP8 · NVFP4 · W4AFP8 · INT8/FP8 paged KV",
    "Anthropic + OpenAI",
  ],
  [
    "<code>metal</code>",
    '<span class="pill warn">beta</span>',
    "Apple Silicon (M1+)",
    "Qwen3.5 / 3.6 MoE (canonical) · Qwen3 dense · DeepSeek-OCR",
    "MLX 4-bit · BF16",
    "Anthropic + OpenAI",
  ],
  [
    "<code>cpu</code>",
    '<span class="pill dim">dev only</span>',
    "portable smoke",
    "Qwen3.5 (small)",
    "BF16",
    "Anthropic + OpenAI",
  ],
];

const MATRIX_ROWS_ZH: string[][] = [
  [
    "<code>cuda</code>",
    '<span class="pill ok">stable</span>',
    "Linux + NVIDIA Ampere+ · 1 到 8 卡（TP / EP）",
    "Qwen3.5 / 3.6 / 3.8 · DeepSeek-V4-Flash · GLM-5.2",
    "BF16 · FP8 · NVFP4 · W4AFP8 · INT8/FP8 分页 KV",
    "Anthropic + OpenAI",
  ],
  [
    "<code>metal</code>",
    '<span class="pill warn">beta</span>',
    "Apple Silicon（M1+）",
    "Qwen3.5 / 3.6 MoE（canonical）· Qwen3 dense · DeepSeek-OCR",
    "MLX 4-bit · BF16",
    "Anthropic + OpenAI",
  ],
  [
    "<code>cpu</code>",
    '<span class="pill dim">dev only</span>',
    "便携冒烟",
    "Qwen3.5（小尺寸）",
    "BF16",
    "Anthropic + OpenAI",
  ],
];

const WHY_CELLS_EN: WhyCell[] = [
  {
    no: "reason · 01",
    title: "Turns stay fast.",
    body:
      `A coding agent re-sends the whole conversation every turn. ARLE keeps the prior turn's KV on the accelerator, shares prefix pages across requests through a radix cache, and prefills only the tokens the turn added. 12 agent-shaped turns on a MacBook: <b>180 ms</b> per turn against mlx-lm's 249 ms on the same weights — <a href="${WINS}/2026-09-02-metal-prefix-restore-survives-turns.md">measured</a>, not projected.`,
  },
  {
    no: "reason · 02",
    title: "Speaks both agent dialects.",
    body:
      "Anthropic <code>/v1/messages</code> with streaming events, <code>tool_use</code> and extended-thinking blocks, and OpenAI <code>/v1/chat/completions</code> with streaming and tools. Claude Code needs one environment variable; any <code>model</code> string is routed to the served model.",
  },
  {
    no: "reason · 03",
    title: "One binary, no Python.",
    body:
      "Pure Rust from the HTTP layer to the scheduler and the KV cache. Metal runs through an MLX C++ bridge, NVIDIA through FlashMLA, DeepGEMM and DeepEP with CUDA graphs. <code>brew install</code>, one curl line, or a Docker image — <b>clone to first token in minutes</b>.",
  },
  {
    no: "reason · 04",
    title: "Numbers are dated.",
    body:
      `Every figure on this page resolves to a dated snapshot in <a href="${GH}/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> with the command, the environment, and the arm it beat. Speculative decode is verified bit-identical to greedy; restored prefixes are gated against cold prefill with a needle ladder.`,
  },
];

const WHY_CELLS_ZH: WhyCell[] = [
  {
    no: "reason · 01",
    title: "多轮不变慢。",
    body:
      `coding agent 每一轮都把整段对话重发一遍。ARLE 把上一轮的 KV 留在加速器上，前缀页经 radix cache 跨请求共享，每轮只 prefill 新增的 token。MacBook 上 12 轮 agent 形状的对话：每轮 <b>180 ms</b>，同一份权重上 mlx-lm 是 249 ms —— <a href="${WINS}/2026-09-02-metal-prefix-restore-survives-turns.md">实测</a>，不是推算。`,
  },
  {
    no: "reason · 02",
    title: "两种 agent 方言都会说。",
    body:
      "Anthropic <code>/v1/messages</code>：流式事件、<code>tool_use</code>、extended thinking 块；OpenAI <code>/v1/chat/completions</code>：流式与工具调用。Claude Code 只需要一个环境变量；任何 <code>model</code> 字串都会路由到当前服务的模型。",
  },
  {
    no: "reason · 03",
    title: "一个二进制，没有 Python。",
    body:
      "从 HTTP 层到调度器到 KV cache 全部是 Rust。Metal 走 MLX C++ bridge，NVIDIA 走 FlashMLA、DeepGEMM、DeepEP 加 CUDA graph。<code>brew install</code>、一行 curl 或一个 Docker 镜像 —— <b>从 clone 到第一个 token 只要几分钟</b>。",
  },
  {
    no: "reason · 04",
    title: "数字都有日期。",
    body:
      `这一页的每个数字都能在 <a href="${GH}/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> 找到带日期的快照：命令、环境、以及它击败的那条臂。推测解码逐比特对齐 greedy；恢复的前缀用 needle 阶梯对照冷 prefill 做门控。`,
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
    note: "Anthropic + OpenAI facade (axum) · <code>Engine&lt;E,K&gt;</code> · scheduler · radix prefix cache",
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
  "<b>train</b> · autograd + train — OPD only; the teacher is the serving engine",
];

const ARCH_FOOT_ZH: string[] = [
  "<b>纯叶子</b> · infer-topo · infer-moe · infer-util",
  "<b>specs</b> · qwen3 · qwen35 · deepseek",
  "<b>ffi</b> · deepep-sys · xgrammar-sys",
  "<b>train</b> · autograd + train —— 仅 OPD；teacher 就是 serving 引擎",
];

const BATTLE_ROWS_EN: BattleRow[] = [
  {
    pri: "P1 · active",
    hot: true,
    title: "Multi-turn TTFT on the 35B",
    desc: "The per-turn table ships on Qwen3.5-0.8B. The canonical Qwen3.6-35B-A3B row needs a MacBook without swap pressure; the script and the method are in the repo",
    where: "scripts/bench_multiturn_ttft.py · benchmarks/",
  },
  {
    pri: "P1 · active",
    title: "Batched speculative verify",
    desc: "Speculative decode is inert from c=4 until the verify step is batched; the measured top lever on the NVFP4 27B, projected −32% decode latency at c=16",
    where: "infer-cuda · docs/plans/2026-08-21-batched-mtp-verify.md",
  },
  {
    pri: "P2 · active",
    title: "Prefill parity with SGLang",
    desc: "Decode is 2.8% faster than SGLang on the same kernel; prefill of a 33K prompt is 19% slower (25.0 s vs 21.0 s)",
    where: "infer-cuda · docs/baselines.md",
  },
  {
    pri: "P2 · queued",
    title: "Cold page-read latency of the KV tiers",
    desc: "Prefix pages demote to host RAM and disk under pressure; a decode step stalling on a cold page is the design risk, and its latency is unmeasured",
    where: "infer-core · kv-native-sys",
  },
  {
    pri: "open",
    title: "Distill your agent traces",
    desc: "On-Policy Distillation from a DeepSeek-V4-Flash teacher into a 35B student on the serving engine; today the teacher runs on CUDA only",
    where: "train · autograd",
  },
  {
    pri: "open",
    title: "Third backends: HIP / Vulkan",
    desc: "HIP substrate and a coherent Vulkan forward (gfx1151) landed; the license is performance parity, not a boot",
    where: "infer-hip · infer-vulkan · #71",
  },
];

const BATTLE_ROWS_ZH: BattleRow[] = [
  {
    pri: "P1 · active",
    hot: true,
    title: "35B 上的多轮 TTFT",
    desc: "每轮 TTFT 表目前是 Qwen3.5-0.8B。canonical 的 Qwen3.6-35B-A3B 那一行需要一台没有 swap 压力的 MacBook；脚本和方法都在仓库里",
    where: "scripts/bench_multiturn_ttft.py · benchmarks/",
  },
  {
    pri: "P1 · active",
    title: "批量化的推测校验",
    desc: "c=4 起推测解码失效，直到 verify 步批量化；NVFP4 27B 上实测的最大杠杆，预计 c=16 解码延迟 −32%",
    where: "infer-cuda · docs/plans/2026-08-21-batched-mtp-verify.md",
  },
  {
    pri: "P2 · active",
    title: "prefill 追平 SGLang",
    desc: "同一 kernel 上解码比 SGLang 快 2.8%；33K 提示的 prefill 慢 19%（25.0 s 对 21.0 s）",
    where: "infer-cuda · docs/baselines.md",
  },
  {
    pri: "P2 · queued",
    title: "KV 分层的冷页读取延迟",
    desc: "前缀页在内存压力下下沉到主机内存和磁盘；解码步在冷页上卡住是设计风险，其延迟尚未测量",
    where: "infer-core · kv-native-sys",
  },
  {
    pri: "open",
    title: "把你的 agent 轨迹蒸馏进小模型",
    desc: "以 DeepSeek-V4-Flash 为 teacher、在 serving 引擎上对 35B student 做 On-Policy Distillation；目前 teacher 只能跑在 CUDA 上",
    where: "train · autograd",
  },
  {
    pri: "open",
    title: "第三后端：HIP / Vulkan",
    desc: "HIP 基板与 coherent 的 Vulkan 前向（gfx1151）已落地；license 条件是性能 parity，不是能跑",
    where: "infer-hip · infer-vulkan · #71",
  },
];

const FILES_EN: FileRow[] = [
  { path: "/README.md", desc: "install · connect an agent · per-turn TTFT table", href: `${GH}/blob/main/README.md` },
  { path: "/docs/http-api.md", desc: "Anthropic + OpenAI routes · streaming behavior", href: `${GH}/blob/main/docs/http-api.md` },
  { path: "/docs/support-matrix.md", desc: "backend / model / quant support", href: `${GH}/blob/main/docs/support-matrix.md` },
  { path: "/docs/baselines.md", desc: "one SOTA row per model, with its config", href: `${GH}/blob/main/docs/baselines.md` },
  { path: "/docs/codebase-map.md", desc: "canonical workspace topology", href: `${GH}/blob/main/docs/codebase-map.md` },
  { path: "/docs/experience/wins/", desc: "dated benchmark snapshots", href: `${GH}/tree/main/docs/experience/wins` },
  { path: "/crates/cli/", desc: "arle binary · verbs · doctor", href: `${GH}/tree/main/crates/cli` },
  { path: "/crates/infer-server/", desc: "/v1/messages · /v1/chat/completions", href: `${GH}/tree/main/crates/infer-server` },
  { path: "/crates/infer-core/", desc: "runtime spine · engine · scheduler · radix cache", href: `${GH}/tree/main/crates/infer-core` },
  { path: "/crates/infer-metal/", desc: "MLX executor · prefix snapshots · KV disk tier", href: `${GH}/tree/main/crates/infer-metal` },
  { path: "/crates/train/", desc: "OPD loop · autograd tape · seq-chunked recompute", href: `${GH}/tree/main/crates/train` },
  { path: "/scripts/bench_multiturn_ttft.py", desc: "the per-turn TTFT measurement", href: `${GH}/blob/main/scripts/bench_multiturn_ttft.py` },
  { path: "/releases", desc: "tagged binaries · checksums", href: `${GH}/releases` },
];

const FILES_ZH: FileRow[] = [
  { path: "/README.zh-CN.md", desc: "安装 · 接入 agent · 每轮 TTFT 表", href: `${GH}/blob/main/README.zh-CN.md` },
  { path: "/docs/http-api.md", desc: "Anthropic + OpenAI 路由 · 流式行为", href: `${GH}/blob/main/docs/http-api.md` },
  { path: "/docs/support-matrix.md", desc: "后端 / 模型 / 量化支持", href: `${GH}/blob/main/docs/support-matrix.md` },
  { path: "/docs/baselines.md", desc: "每个模型一行 SOTA 及其配置", href: `${GH}/blob/main/docs/baselines.md` },
  { path: "/docs/codebase-map.md", desc: "权威 workspace 拓扑", href: `${GH}/blob/main/docs/codebase-map.md` },
  { path: "/docs/experience/wins/", desc: "带日期的基准快照", href: `${GH}/tree/main/docs/experience/wins` },
  { path: "/crates/cli/", desc: "arle 二进制 · 子命令 · doctor", href: `${GH}/tree/main/crates/cli` },
  { path: "/crates/infer-server/", desc: "/v1/messages · /v1/chat/completions", href: `${GH}/tree/main/crates/infer-server` },
  { path: "/crates/infer-core/", desc: "运行时主干 · engine · scheduler · radix cache", href: `${GH}/tree/main/crates/infer-core` },
  { path: "/crates/infer-metal/", desc: "MLX 执行器 · 前缀快照 · KV 磁盘层", href: `${GH}/tree/main/crates/infer-metal` },
  { path: "/crates/train/", desc: "OPD 循环 · autograd tape · seq-chunked recompute", href: `${GH}/tree/main/crates/train` },
  { path: "/scripts/bench_multiturn_ttft.py", desc: "每轮 TTFT 的测量脚本", href: `${GH}/blob/main/scripts/bench_multiturn_ttft.py` },
  { path: "/releases", desc: "发版二进制 · 校验和", href: `${GH}/releases` },
];

export const EN: Locale = {
  lang: "en",
  hreflang: "en",
  meta: {
    title: "arle — the local inference server for coding agents",
    description:
      "ARLE is a pure-Rust inference server for Apple Silicon and NVIDIA. Anthropic and OpenAI APIs in one binary, a KV cache that survives across turns, so Claude Code or any OpenAI-API agent on a local model starts turn 20 as fast as turn 2.",
    ogTitle: "arle — point your coding agent at a local model",
    ogDescription:
      "Pure Rust, one binary, Apple Silicon and NVIDIA. Anthropic + OpenAI APIs. The KV cache survives across turns: 180 ms per turn on a MacBook against mlx-lm's 249 ms.",
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
      { label: "github ↗", href: GH },
    ],
  },
  hero: {
    kicker: "pure rust · one binary · apple silicon + nvidia · anthropic + openai apis",
    headline:
      'Point your coding agent<br>at a <span class="magic">local model</span><span class="quiet">.</span>',
    lede:
      "Claude Code, opencode, aider: every turn re-sends the whole conversation. ARLE keeps the KV cache alive across turns and prefills only what the turn added — <b>turn 20 starts as fast as turn 2</b>.",
    signals: SIGNALS,
    ctas: [
      { label: "$ Quickstart", href: "#install" },
      { label: "★ Star acupof-ai/arle", href: GH },
      { label: "HTTP API →", href: `${GH}/blob/main/docs/http-api.md` },
    ],
    terminal: {
      title: "arle — zsh",
      cwd: "~/code",
      lines: TERMINAL_LINES_EN,
    },
  },
  sections: {
    why: {
      title: "Why this exists",
      caption:
        "Most local servers are built around one prompt. An agent session is a loop: the same system prompt and a growing tool history, twenty times over. ARLE is built around the loop.",
      cells: WHY_CELLS_EN,
    },
    architecture: {
      title: "Architecture",
      caption:
        `One runtime, three surfaces, two backends. Serving, the local agent, and OPD training run the same Rust and model code; dependencies flow strictly downward and <b>infer-core carries no backend dependency</b>. Canonical topology lives in <a href="${GH}/blob/main/docs/codebase-map.md"><code>docs/codebase-map.md</code></a>.`,
      rows: ARCH_ROWS,
      foot: ARCH_FOOT_EN,
    },
    install: {
      title: "Install and connect",
      caption:
        `One runnable line per platform, then one environment variable for your agent. Pre-built tarballs and SHAs on each <a href="${GH}/releases">GitHub Release</a>; the curl installer verifies SHA256 before extracting.`,
      cards: INSTALL_CARDS_EN,
    },
    bench: {
      title: "Bench",
      caption:
        `Dated, reproducible snapshots straight from <a href="${GH}/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> and <a href="${GH}/blob/main/benchmarks/README.md"><code>benchmarks/</code></a>. Decode and prefill are reported separately; nothing is an end-to-end blend.`,
      rows: BENCH_ROWS_EN,
    },
    matrix: {
      title: "Support matrix",
      caption:
        `Two backends, one runtime contract. Authoritative truth lives in <a href="${GH}/blob/main/docs/support-matrix.md"><code>docs/support-matrix.md</code></a>.`,
      head: ["backend", "stability", "os / hardware", "models", "quants", "api"],
      rows: MATRIX_ROWS_EN,
    },
    contribute: {
      title: "Where a contribution lands",
      caption:
        `No queue, no committee — <b>a weekend PR here can move a headline number</b>, and the fronts are public. Start with <a href="${GH}/blob/main/CONTRIBUTING.md"><code>CONTRIBUTING.md</code></a>.`,
      rows: BATTLE_ROWS_EN,
      starAsk: {
        html:
          "<b>Stars are the only metric a small project has.</b> If ARLE made your agent loop faster than what you had, leave one. It decides how much time this gets.",
        cta: { label: "★ Star acupof-ai/arle", href: GH },
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
    left: "arle(1) · September 2026 · v0.5.8",
    right: { label: "github.com/acupof-ai/arle", href: GH },
  },
};

export const ZH: Locale = {
  lang: "zh-Hans",
  hreflang: "zh-Hans",
  meta: {
    title: "arle — 给 coding agent 用的本地推理服务器",
    description:
      "ARLE 是纯 Rust 推理服务器，跑在 Apple Silicon 与 NVIDIA 上。一个二进制同时提供 Anthropic 与 OpenAI API，KV cache 跨轮常驻，Claude Code 或任何 OpenAI API agent 接本地模型，第 20 轮和第 2 轮一样快。",
    ogTitle: "arle — 把你的 coding agent 指向本地模型",
    ogDescription:
      "纯 Rust，单个二进制，Apple Silicon 与 NVIDIA。Anthropic + OpenAI API。KV cache 跨轮常驻：MacBook 上每轮 180 ms，mlx-lm 是 249 ms。",
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
      { label: "github ↗", href: GH },
    ],
  },
  hero: {
    kicker: "纯 rust · 单个二进制 · apple silicon + nvidia · anthropic + openai api",
    headline:
      '把你的 coding agent<br>指向<span class="magic">本地模型</span><span class="quiet">。</span>',
    lede:
      "Claude Code、opencode、aider：每一轮都把整段对话重发一遍。ARLE 让 KV cache 跨轮常驻，每轮只 prefill 新增的部分 —— <b>第 20 轮和第 2 轮一样快</b>。",
    signals: SIGNALS,
    ctas: [
      { label: "$ 快速开始", href: "#install" },
      { label: "★ Star acupof-ai/arle", href: GH },
      { label: "HTTP API →", href: `${GH}/blob/main/docs/http-api.md` },
    ],
    terminal: {
      title: "arle — zsh",
      cwd: "~/code",
      lines: TERMINAL_LINES_EN,
    },
  },
  sections: {
    why: {
      title: "为什么做这个",
      caption:
        "多数本地服务器是围着单条 prompt 设计的。agent 会话是一个循环：同一段系统提示加上越来越长的工具历史，重复二十遍。ARLE 围着这个循环设计。",
      cells: WHY_CELLS_ZH,
    },
    architecture: {
      title: "架构",
      caption:
        `一套运行时、三个表面、两个后端。serving、本地 agent、OPD 训练跑同一份 Rust 与模型代码；依赖严格向下流动，<b>infer-core 不依赖任何后端</b>。权威拓扑见 <a href="${GH}/blob/main/docs/codebase-map.md"><code>docs/codebase-map.md</code></a>。`,
      rows: ARCH_ROWS,
      foot: ARCH_FOOT_ZH,
    },
    install: {
      title: "安装与接入",
      caption:
        `每个平台一行能跑的命令，然后给 agent 一个环境变量。预编译 tarball 与 SHA 见每次 <a href="${GH}/releases">GitHub Release</a>；curl 安装脚本会先校验 SHA256 再解压。`,
      cards: INSTALL_CARDS_ZH,
    },
    bench: {
      title: "基准",
      caption:
        `直接来自 <a href="${GH}/tree/main/docs/experience/wins"><code>docs/experience/wins/</code></a> 与 <a href="${GH}/blob/main/benchmarks/README.md"><code>benchmarks/</code></a> 的带日期快照。解码与 prefill 分开报告，没有端到端混合数。`,
      rows: BENCH_ROWS_ZH,
    },
    matrix: {
      title: "支持矩阵",
      caption:
        `两种后端，一份运行时契约。权威矩阵见 <a href="${GH}/blob/main/docs/support-matrix.md"><code>docs/support-matrix.md</code></a>。`,
      head: ["后端", "稳定度", "系统 / 硬件", "模型", "量化", "API"],
      rows: MATRIX_ROWS_ZH,
    },
    contribute: {
      title: "贡献会落在哪",
      caption:
        `没有排队，没有委员会 —— <b>一个周末的 PR 就能改动头条数字</b>，战线全部公开。从 <a href="${GH}/blob/main/CONTRIBUTING.md"><code>CONTRIBUTING.md</code></a> 开始。`,
      rows: BATTLE_ROWS_ZH,
      starAsk: {
        html:
          "<b>Star 是小项目唯一的指标。</b>如果 ARLE 让你的 agent 循环比原来快了，留一颗。它决定这件事能得到多少时间。",
        cta: { label: "★ Star acupof-ai/arle", href: GH },
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
    left: "arle(1) · 2026 年 9 月 · v0.5.8",
    right: { label: "github.com/acupof-ai/arle", href: GH },
  },
};
