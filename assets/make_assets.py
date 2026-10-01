#!/usr/bin/env python3
"""Regenerate the README header image in light and dark. Stdlib only."""
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))

THEMES = {
    "dark": dict(bg="#0d1117", panel="#161b22", line="#30363d", text="#e6edf3", dim="#8b949e",
                 accent="#a371f7"),
    "light": dict(bg="#ffffff", panel="#f6f8fa", line="#d0d7de", text="#1f2328", dim="#656d76",
                  accent="#8250df"),
}
FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
ALT = ("agnes-nvfp4: Agnes-3.0-Flash bf16, fold to a stock Qwen3.5 graph, quantize to NVFP4, "
       "re-attach the bf16 MTP head, serve in stock vLLM")


def box(x, y, w, h, label, subs, c, stroke):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{c["panel"]}" stroke="{stroke}" stroke-width="1.5"/>',
         f'<text x="{x + w / 2}" y="{y + 30}" text-anchor="middle" font-family="{FONT}" font-size="17" font-weight="600" fill="{c["text"]}">{html.escape(label)}</text>']
    for i, (sub, mono) in enumerate(subs):
        fam, size = (MONO, 11.5) if mono else (FONT, 12.5)
        s.append(f'<text x="{x + w / 2}" y="{y + 54 + i * 19}" text-anchor="middle" font-family="{fam}" font-size="{size}" fill="{c["dim"]}">{html.escape(sub)}</text>')
    return "".join(s)


def arrow(x1, y1, x2, y2, color, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 7}" y2="{y2}" stroke="{color}" stroke-width="1.8"{dash}/>'
            f'<path d="M{x2 - 8},{y2 - 5} L{x2},{y2} L{x2 - 8},{y2 + 5} Z" fill="{color}"/>')


def hero(c):
    W, H = 1200, 330
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="{html.escape(ALT)}">',
         f'<rect width="{W}" height="{H}" rx="16" fill="{c["bg"]}" stroke="{c["line"]}"/>',
         f'<text x="60" y="78" font-family="{MONO}" font-size="38" font-weight="700" fill="{c["text"]}">agnes-nvfp4</text>',
         f'<text x="60" y="112" font-family="{FONT}" font-size="18" fill="{c["dim"]}">'
         'Agnes-3.0-Flash Preview, folded to a stock Qwen3.5 graph and quantized to NVFP4 for one DGX Spark.</text>']
    y, w, h, gap = 150, 184, 96, 40
    nodes = [
        ("Agnes-3.0-Flash", [("bf16 Preview checkpoint", False), ("72 layers + parallel_ffn", False)]),
        ("fold", [("fold_agnes_to_qwen35.py", True), ("MLP 17408 + 2048 = 19456", False)]),
        ("quantize", [("quant_agnes_nvfp4.py", True), ("NVFP4 W4A4, llm-compressor", False)]),
        ("MTP head", [("wire-mtp-into-nvfp4.py", True), ("kept bf16, padded to fit", False)]),
        ("stock vLLM", [("Qwen3_5 architecture", False), ("no trust_remote_code", False)]),
    ]
    for i, (label, subs) in enumerate(nodes):
        x = 60 + i * (w + gap)
        s.append(box(x, y, w, h, label, subs, c, c["accent"] if i == 4 else c["line"]))
        if i < len(nodes) - 1:
            s.append(arrow(x + w, y + h / 2, x + w + gap, y + h / 2, c["dim"]))
    # verification side path under the fold step
    fx = 60 + (w + gap) + w / 2
    s.append(f'<line x1="{fx}" y1="{y + h}" x2="{fx}" y2="{y + h + 30}" stroke="{c["accent"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    s.append(f'<circle cx="{fx}" cy="{y + h + 34}" r="4" fill="{c["accent"]}"/>')
    s.append(f'<text x="{fx + 14}" y="{y + h + 39}" font-family="{FONT}" font-size="13.5" fill="{c["dim"]}">'
             'checked against the original modeling code: teacher-forced logits, '
             f'<tspan font-family="{MONO}" font-size="12.5" fill="{c["text"]}">logits_probe.py</tspan> + '
             f'<tspan font-family="{MONO}" font-size="12.5" fill="{c["text"]}">compare_logits.py</tspan></text>')
    s.append("</svg>")
    return "".join(s)


for theme, colors in THEMES.items():
    with open(os.path.join(HERE, f"hero-{theme}.svg"), "w", encoding="utf-8") as f:
        f.write(hero(colors))
print("ok")
