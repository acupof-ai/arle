#!/usr/bin/env python3
"""Generate rich SVG charts for DSv4 / GLM-5.2 MoE SVD analysis.

Outputs SVG files that can be uploaded to Lark whiteboard.
Charts: spectrum/frame/merge line charts, cluster heatmap, MDS scatter,
and cross-model comparison.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# Color palette
COLORS = {
    "gate": "#e74c3c",
    "down": "#3498db",
    "up": "#2ecc71",
    "dsv4": "#e74c3c",
    "glm52": "#9b59b6",
}
GRID = "#e8e8e8"
AXIS = "#333"
BG = "#ffffff"


def svg_header(w, h, title=None):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>')
    if title:
        parts.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold" fill="#222">{title}</text>')
    return parts


def line_chart(data, title, ylabel, ymin=None, ymax=None, w=900, h=420):
    """data: dict of series_name -> list of (x, y) or list of y (x=index)."""
    ml, mr, mt, mb = 70, 30, 60, 60
    plot_w = w - ml - mr
    plot_h = h - mt - mb

    all_y = [v for series in data.values() for v in series]
    if ymin is None:
        ymin = min(all_y)
    if ymax is None:
        ymax = max(all_y)
    ypad = (ymax - ymin) * 0.08 if ymax > ymin else 0.1
    ymin -= ypad
    ymax += ypad

    n = max(len(s) for s in data.values())
    xs = list(range(n))

    def sx(i):
        return ml + (i / (n - 1)) * plot_w if n > 1 else ml

    def sy(v):
        return mt + plot_h - (v - ymin) / (ymax - ymin) * plot_h

    svg = svg_header(w, h, title)
    # grid + y axis labels
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        y = sy(yv)
        svg.append(f'<line x1="{ml}" y1="{y}" x2="{ml+plot_w}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{ml-8}" y="{y+4}" text-anchor="end" font-size="11" fill="#666">{yv:.3f}</text>')
    # x axis labels (every ~5 layers)
    step = max(1, n // 9)
    for i in range(0, n, step):
        x = sx(i)
        svg.append(f'<line x1="{x}" y1="{mt+plot_h}" x2="{x}" y2="{mt+plot_h+5}" stroke="{AXIS}" stroke-width="1"/>')
        svg.append(f'<text x="{x}" y="{mt+plot_h+18}" text-anchor="middle" font-size="11" fill="#666">{i}</text>')
    # axis lines
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+plot_h}" stroke="{AXIS}" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" stroke="{AXIS}" stroke-width="1.5"/>')
    # axis labels
    svg.append(f'<text x="{ml+plot_w/2}" y="{h-15}" text-anchor="middle" font-size="13" fill="#333">层</text>')
    svg.append(f'<text x="18" y="{mt+plot_h/2}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90,18,{mt+plot_h/2})">{ylabel}</text>')
    # legend
    lx = ml + 10
    for name, series in data.items():
        color = COLORS.get(name, "#333")
        svg.append(f'<rect x="{lx}" y="{mt-22}" width="14" height="3" fill="{color}"/>')
        svg.append(f'<text x="{lx+18}" y="{mt-18}" font-size="12" fill="#333">{name}</text>')
        lx += len(name) * 12 + 50
    # lines + points
    for name, series in data.items():
        color = COLORS.get(name, "#333")
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(series))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in enumerate(series):
            svg.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="2.5" fill="{color}">')
            svg.append(f'<title>{name} layer={i} value={v:.4f}</title></circle>')
    svg.append("</svg>")
    return "\n".join(svg)


def heatmap(data, layers, ks, title, w=920, h=300):
    """data: dict k_str -> list of ratio per layer."""
    ml, mr, mt, mb = 65, 120, 60, 70
    cell_w = (w - ml - mr) / len(layers)
    cell_h = (h - mt - mb) / len(ks)

    def color(r):
        t = max(0, min(1, (r - 1.0) / 0.8))
        return f"rgb(255,{int(255*(1-t))},{int(255*(1-t))})"

    svg = svg_header(w, h, title)
    # y labels
    for i, k in enumerate(ks):
        y = mt + i * cell_h + cell_h / 2
        svg.append(f'<text x="{ml-8}" y="{y+4}" text-anchor="end" font-size="12" fill="#333">k={k}</text>')
    # x labels (every 5)
    for j, L in enumerate(layers):
        if L % 5 == 0:
            x = ml + j * cell_w + cell_w / 2
            svg.append(f'<text x="{x}" y="{mt+len(ks)*cell_h+15}" text-anchor="middle" font-size="10" fill="#666">{L}</text>')
    # cells
    for i, k in enumerate(ks):
        for j, L in enumerate(layers):
            r = data[str(k)][j]
            x = ml + j * cell_w
            y = mt + i * cell_h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color(r)}" stroke="#eee" stroke-width="0.5">')
            svg.append(f'<title>layer={L} k={k} ratio={r:.3f}</title></rect>')
            if r > 1.25 and cell_w > 12:
                svg.append(f'<text x="{x+cell_w/2:.1f}" y="{y+cell_h/2+3:.1f}" text-anchor="middle" font-size="8" fill="#000">{r:.2f}</text>')
    # color bar (solid steps, no gradient — whiteboard doesn't support gradients)
    bx = ml + len(layers) * cell_w + 20
    by = mt
    bh = len(ks) * cell_h
    steps = 8
    for s in range(steps):
        r = 1.0 + 0.8 * s / (steps - 1)
        sy0 = by + bh * (1 - (s + 1) / steps)
        sh = bh / steps
        svg.append(f'<rect x="{bx}" y="{sy0:.1f}" width="16" height="{sh:.1f}" fill="{color(r)}" stroke="#eee" stroke-width="0.5"/>')
    svg.append(f'<rect x="{bx}" y="{by}" width="16" height="{bh}" fill="none" stroke="#999"/>')
    svg.append(f'<text x="{bx+22}" y="{by+8}" font-size="10" fill="#666">1.8</text>')
    svg.append(f'<text x="{bx+22}" y="{by+bh}" font-size="10" fill="#666">1.0</text>')
    svg.append(f'<text x="{bx-5}" y="{by+bh+18}" font-size="10" fill="#666" text-anchor="end">ratio</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def scatter_grid(scatter_data, layers, title, w=1200, h=320):
    """scatter_data: dict layer_str -> {coords, labels}."""
    ml, mr, mt, mb = 40, 20, 60, 50
    gap = 25
    n = len(layers)
    sub_w = (w - ml - mr - gap * (n - 1)) / n
    sub_h = h - mt - mb
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    svg = svg_header(w, h, title)
    for idx, sl in enumerate(layers):
        d = scatter_data[str(sl)]
        coords = d["coords"]
        labels = d["labels"]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xr = xmax - xmin if xmax > xmin else 1
        yr = ymax - ymin if ymax > ymin else 1
        ox = ml + idx * (sub_w + gap)
        oy = mt
        svg.append(f'<text x="{ox+sub_w/2}" y="{oy-10}" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Layer {sl}</text>')
        svg.append(f'<rect x="{ox}" y="{oy}" width="{sub_w}" height="{sub_h}" fill="#fafafa" stroke="#ccc"/>')
        for ci, (x, y) in enumerate(coords):
            px = ox + 8 + (x - xmin) / xr * (sub_w - 16)
            py = oy + 8 + (y - ymin) / yr * (sub_h - 16)
            c = palette[labels[ci] % len(palette)]
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{c}" opacity="0.7">')
            svg.append(f'<title>expert={ci} cluster={labels[ci]}</title></circle>')
    svg.append("</svg>")
    return "\n".join(svg)


def comparison_chart(dsv4, glm52, key, title, ylabel, w=900, h=420):
    """Compare one metric across two models, per layer, averaged over proj kinds."""
    layers = sorted(dsv4["per_layer"].keys(), key=int)
    series = {}
    for model_name, model_data, prefix in [("DSv4", dsv4, "dsv4"), ("GLM-5.2", glm52, "glm52")]:
        for kind in ["gate_proj", "down_proj", "up_proj"]:
            vals = []
            for L in layers:
                if L in model_data["per_layer"] and kind in model_data["per_layer"][L]:
                    vals.append(model_data["per_layer"][L][kind][key])
                else:
                    vals.append(None)
            if any(v is not None for v in vals):
                series[f"{model_name}-{kind}"] = vals
    # simplify: just plot gate_proj for both models
    simple = {}
    for mn, md in [("DSv4", dsv4), ("GLM-5.2", glm52)]:
        simple[mn] = [md["per_layer"][L]["gate_proj"][key] for L in layers if L in md.get("per_layer", {})]
    return line_chart(simple, title, ylabel, w=w, h=h)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsv4", required=True, help="DSv4 fast result JSON")
    ap.add_argument("--dsv4-cluster", help="DSv4 cluster JSON")
    ap.add_argument("--glm52", help="GLM-5.2 fast result JSON")
    ap.add_argument("--glm52-cluster", help="GLM-5.2 cluster JSON")
    ap.add_argument("--outdir", default="/tmp/charts", help="output dir")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    dsv4 = json.load(open(args.dsv4))
    layers = sorted(dsv4["per_layer"].keys(), key=int)
    L = [int(x) for x in layers]

    # 1. spectrum similarity
    data = {
        "gate": [dsv4["per_layer"][x]["gate_proj"]["spec_cos_mean"] for x in layers],
        "down": [dsv4["per_layer"][x]["down_proj"]["spec_cos_mean"] for x in layers],
        "up": [dsv4["per_layer"][x]["up_proj"]["spec_cos_mean"] for x in layers],
    }
    (out / "01_spec.svg").write_text(line_chart(data, "谱相似度随深度变化", "余弦相似度"))

    # 2. frame similarity
    data = {
        "gate": [dsv4["per_layer"][x]["gate_proj"]["frame_sim_u"] for x in layers],
        "down": [dsv4["per_layer"][x]["down_proj"]["frame_sim_u"] for x in layers],
        "up": [dsv4["per_layer"][x]["up_proj"]["frame_sim_u"] for x in layers],
    }
    (out / "02_frame.svg").write_text(line_chart(data, "帧相似度随深度变化 (U空间)", "主角度余弦"))

    # 3. merge score
    data = {
        "gate": [dsv4["per_layer"][x]["gate_proj"]["spec_cos_mean"] * (1 - dsv4["per_layer"][x]["gate_proj"]["frame_sim_u"]) for x in layers],
        "down": [dsv4["per_layer"][x]["down_proj"]["spec_cos_mean"] * (1 - dsv4["per_layer"][x]["down_proj"]["frame_sim_u"]) for x in layers],
        "up": [dsv4["per_layer"][x]["up_proj"]["spec_cos_mean"] * (1 - dsv4["per_layer"][x]["up_proj"]["frame_sim_u"]) for x in layers],
    }
    (out / "03_merge.svg").write_text(line_chart(data, "合并得分随深度变化", "merge_score"))

    # 4. cluster heatmap
    if args.dsv4_cluster:
        cl = json.load(open(args.dsv4_cluster))
        (out / "04_heatmap.svg").write_text(
            heatmap(cl["heatmap"]["gate_proj"], cl["heatmap"]["layers"], cl["heatmap"]["ks"],
                    "gate_proj 聚类 ratio 热力图 (全层 × k=2,4,8,16)"))

    # 5. scatter
    if args.dsv4_cluster:
        cl = json.load(open(args.dsv4_cluster))
        scatter_layers = [int(x) for x in cl["scatter"].keys()]
        (out / "05_scatter.svg").write_text(
            scatter_grid(cl["scatter"], scatter_layers, "gate_proj 专家帧空间 MDS 散点图 (k=4 聚类着色)"))

    # 6. comparison with GLM-5.2
    if args.glm52:
        glm = json.load(open(args.glm52))
        glm_layers = sorted(glm["per_layer"].keys(), key=int)
        # spectrum comparison (gate_proj)
        d1 = {"DSv4": [dsv4["per_layer"][x]["gate_proj"]["spec_cos_mean"] for x in layers],
              "GLM-5.2": [glm["per_layer"][x]["gate_proj"]["spec_cos_mean"] for x in glm_layers if x in glm["per_layer"]]}
        (out / "06_cmp_spec.svg").write_text(line_chart(d1, "DSv4 vs GLM-5.2 谱相似度对比 (gate_proj)", "余弦相似度"))
        # frame comparison
        d2 = {"DSv4": [dsv4["per_layer"][x]["gate_proj"]["frame_sim_u"] for x in layers],
              "GLM-5.2": [glm["per_layer"][x]["gate_proj"]["frame_sim_u"] for x in glm_layers if x in glm["per_layer"]]}
        (out / "07_cmp_frame.svg").write_text(line_chart(d2, "DSv4 vs GLM-5.2 帧相似度对比 (gate_proj)", "主角度余弦"))
        # merge score comparison
        d3 = {"DSv4": [dsv4["per_layer"][x]["gate_proj"]["spec_cos_mean"]*(1-dsv4["per_layer"][x]["gate_proj"]["frame_sim_u"]) for x in layers],
              "GLM-5.2": [glm["per_layer"][x]["gate_proj"]["spec_cos_mean"]*(1-glm["per_layer"][x]["gate_proj"]["frame_sim_u"]) for x in glm_layers if x in glm["per_layer"]]}
        (out / "08_cmp_merge.svg").write_text(line_chart(d3, "DSv4 vs GLM-5.2 合并得分对比 (gate_proj)", "merge_score"))

    print(f"Charts written to {out}")


if __name__ == "__main__":
    main()
