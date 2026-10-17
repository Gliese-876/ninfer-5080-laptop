#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the public throughput chart from the adjacent measured data.

Requires Python 3.10+, Matplotlib 3.8+, and the packaged Noto Sans fonts.
Run: python figure_sources/plot_throughput.py
Font sources and the SIL Open Font License are included in fonts/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT_FAMILY = "Noto Sans"
FONT_STACK = [FONT_FAMILY]
FONT_FILES = {"regular": "NotoSans-Regular.ttf", "bold": "NotoSans-Bold.ttf"}
WIDTH_PX, HEIGHT_PX, DPI = 1200, 700, 96
CHART_TITLE = "Qwen3.8-27B · 118K Decode Throughput"
GROUP_ORDER = ["desktop_original", "laptop_baseline", "laptop_optimized"]
COLORS = ["#BFC5CE", "#8993A3", "#1769E8"]
TEXT, MUTED, GRID = "#172132", "#566173", "#E8ECF1"


def resolve_fonts(font_directory: Path | None = None) -> dict:
    """Register the packaged regular and bold faces for deterministic rendering."""
    directory = font_directory or Path(__file__).resolve().parent / "fonts"
    font_manager.fontManager.ttflist = [
        entry for entry in font_manager.fontManager.ttflist if entry.name != FONT_FAMILY
    ]
    faces = {}
    for role, filename in FONT_FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Noto Sans font file is missing: {filename}")
        face = FT2Font(str(path))
        if face.family_name != FONT_FAMILY or face.style_name.casefold() != role:
            raise ValueError(f"Font metadata does not match Noto Sans {role}: {filename}")
        font_manager.fontManager.addfont(str(path))
        faces[role] = {"family": FONT_FAMILY, "file": filename, "style": face.style_name,
                       "weight": 700 if role == "bold" else 400,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    font_manager.fontManager._findfont_cached.cache_clear()
    return {"family": FONT_FAMILY, "faces": faces, "cjk_text_present": False,
            "standalone_font_files_distributed": True, "license": "SIL Open Font License 1.1",
            "source_record": "fonts/SOURCE.json"}


def load_data(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    bars = data.get("bars", [])
    if [item.get("id") for item in bars] != GROUP_ORDER:
        raise ValueError("Data must contain exactly the three declared groups, in order.")
    for item in bars:
        value = item.get("tokens_per_second")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Each group requires a positive measured tokens_per_second value.")
        if not item.get("hardware_label") or not item.get("configuration_label"):
            raise ValueError("Each group requires hardware and configuration labels.")
        if not (item.get("source") or item.get("source_url")
                or (item.get("benchmark_report_sha256") and item.get("source_metric"))):
            raise ValueError("Each group requires its authoritative source.")
    if not data.get("metric"):
        raise ValueError("Declare the measurement scope in metric.")
    return data


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": FONT_STACK, "font.sans-serif": FONT_STACK,
        "text.parse_math": False,
        "font.size": 15, "text.color": TEXT, "axes.linewidth": .85,
        "lines.linewidth": 1.35, "grid.linewidth": .55,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": None, "svg.fonttype": "none", "svg.hashsalt": "ninfer-throughput-v1",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def make_figure(data: dict, fonts: dict):
    configure_style()
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH_PX), ylim=(0, HEIGHT_PX))
    ax.set_axis_off()
    baseline, top = 150., 594.
    centers, bar_width = [310., 650., 990.], 168.
    values = [item["tokens_per_second"] for item in data["bars"]]
    ymax = math.ceil(max(values) * 1.2 / 20) * 20
    # Use few grid lines; the common origin and numeric labels carry the comparison.
    tick_step = 20 if ymax <= 100 else 40
    scale = (top - baseline) / ymax
    title = fig.suptitle(CHART_TITLE, x=111 / WIDTH_PX, y=662 / HEIGHT_PX,
                         ha="left", va="center", fontsize=24, fontweight="bold", color=TEXT)
    texts, rectangles, grid_specs = [title], [], []
    for tick in range(0, int(ymax) + 1, tick_step):
        y = baseline + tick * scale
        grid_specs.append((y, tick))
        texts.append(ax.text(91, y, str(tick), ha="right", va="center", fontsize=14.25, color=MUTED))
    texts.append(ax.text(111, 618, "tok/s", ha="left", va="center", fontsize=16.5, color=MUTED))
    for index, (item, x, value, color) in enumerate(zip(data["bars"], centers, values, COLORS)):
        height = value * scale
        bar = FancyBboxPatch((x - bar_width / 2, baseline), bar_width, height,
                             boxstyle="round,pad=0,rounding_size=12", linewidth=0,
                             facecolor=color, zorder=2)
        bar.set_gid(item["id"])
        ax.add_patch(bar)
        rectangles.append(bar)
        texts.append(ax.text(x, baseline + height + 19, f"{value:.2f}", ha="center", va="bottom",
                             fontsize=22.5, fontweight="bold", color=color if index == 2 else TEXT,
                             zorder=5))
        texts.append(ax.text(x, 101, item["hardware_label"], ha="center", va="center", fontsize=18,
                             color=TEXT))
        texts.append(ax.text(x, 69, item["configuration_label"], ha="center", va="center", fontsize=15,
                             color=color if index == 2 else MUTED,
                             fontweight="bold" if index == 2 else "normal"))
    gain = (values[2] / values[1] - 1) * 100
    arrow_start = (centers[1] + bar_width / 2 + 13, baseline + values[1] * scale + 16)
    arrow_end = (centers[2] - bar_width / 2 - 13, baseline + values[2] * scale + 16)
    arrow = FancyArrowPatch(arrow_start, arrow_end, arrowstyle="-|>", mutation_scale=16,
                            connectionstyle="arc3,rad=-0.38", lw=2.0,
                            color=COLORS[2], shrinkA=0, shrinkB=0, zorder=4)
    arrow.set_gid("laptop-throughput-improvement")
    ax.add_patch(arrow)
    texts.append(ax.text((centers[1] + centers[2]) / 2, max(arrow_start[1], arrow_end[1]) + 56,
                         f"{gain:+.1f}%", ha="center", va="center", fontsize=19.5,
                         fontweight="bold", color=COLORS[2], zorder=5))
    characters = {char for text in texts for char in text.get_text() if not char.isspace()}
    for weight in ["normal", "bold"]:
        font_path = font_manager.findfont(
            font_manager.FontProperties(family=FONT_FAMILY, weight=weight), fallback_to_default=False)
        coverage = FT2Font(font_path).get_charmap()
        missing = sorted(char for char in characters if ord(char) not in coverage)
        if missing:
            raise RuntimeError(f"Noto Sans {weight} lacks required glyphs: {missing!r}")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*Glyph.*missing.*")
        fig.canvas.draw()
    # Split a grid line around any value label instead of drawing through text.
    # Coordinates share the 1200 x 700 pixel canvas, so no crop-dependent offset
    # or extra white label card is needed.
    renderer = fig.canvas.get_renderer()
    text_boxes = [text.get_window_extent(renderer).padded(7) for text in texts]
    for y, tick in grid_specs:
        intervals = [(111., 1144.)]
        for box in text_boxes:
            if not box.y0 <= y <= box.y1:
                continue
            clipped = []
            for start, end in intervals:
                if box.x1 <= start or box.x0 >= end:
                    clipped.append((start, end))
                else:
                    if box.x0 > start:
                        clipped.append((start, box.x0))
                    if box.x1 < end:
                        clipped.append((box.x1, end))
            intervals = clipped
        for start, end in intervals:
            ax.plot([start, end], [y, y], color=GRID if tick else "#CCD3DC",
                    lw=.8 if tick else 1.0, zorder=0)
    fig.canvas.draw()
    contract = {
        "mode": "showcase", "canvas_pixels": [WIDTH_PX, HEIGHT_PX], "dpi": DPI,
        "title_present": True, "title": CHART_TITLE,
        "canvas_inches": [WIDTH_PX / DPI, HEIGHT_PX / DPI], "background": "white",
        "group_order": GROUP_ORDER, "bar_geometry": "rounded rectangles; 12 px corner radius",
        "palette": dict(zip(GROUP_ORDER, COLORS)), "common_zero_origin": True,
        "annotation": {"contrast": "laptop_optimized / laptop_baseline - 1",
                       "gain_percent": gain, "curve_arrow": True},
        "minimum_declared_text_pt": 14.25, "fonts": fonts,
        "svg_text": "live editable text", "svg_raster_images": 0,
        "uncertainty_bars": "none; source supplies point estimates",
    }
    return fig, contract, {"texts": texts, "bars": rectangles, "arrow": arrow}


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "data.json")
    parser.add_argument("--output-directory", type=Path, default=here.parent / "assets")
    parser.add_argument("--qa-directory", type=Path, default=here.parent / "figure-qa")
    parser.add_argument("--font-directory", type=Path)
    args = parser.parse_args()
    data = load_data(args.data)
    fonts = resolve_fonts(args.font_directory)
    fig, contract, _ = make_figure(data, fonts)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.qa_directory.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "pdf", "png"]:
        metadata = {"Creator": "Portable Matplotlib throughput figure"}
        if suffix in {"svg", "pdf"}:
            metadata["Date" if suffix == "svg" else "CreationDate"] = None
        output_path = args.output_directory / f"throughput-comparison.{suffix}"
        fig.savefig(output_path, metadata=metadata)
        if suffix == "svg":
            # Keep live vector text and normalize generated XML whitespace.
            svg = output_path.read_text(encoding="utf-8")
            svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
            output_path.write_bytes(svg.encode("utf-8"))
    contract["source_data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    canonical_data = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract["source_data_canonical_json_sha256"] = hashlib.sha256(canonical_data).hexdigest()
    contract["canonical_json_digest"] = "UTF-8; sorted keys; compact separators; Unicode preserved"
    (args.qa_directory / "figure-contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"outputs": ["throughput-comparison.svg", "throughput-comparison.pdf", "throughput-comparison.png"],
                      "gain_percent": contract["annotation"]["gain_percent"], "font": fonts["family"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
