# Throughput figure

`plot_throughput.py` renders the three point estimates in `data.json` as an editable SVG, a PDF with embedded fonts and a 1200 × 700 px PNG. The plotted contrast is the optimized laptop rate divided by the laptop baseline rate, minus one.

From the repository root, with a Python environment activated:

```bash
python3 -m pip install -r figure_sources/requirements.txt
python3 figure_sources/plot_throughput.py \
  --output-directory assets \
  --qa-directory figure-qa
```

The figure uses a common zero origin, rounded rectangles, two gray reference groups and one blue optimized group. All group labels and values remain live SVG text. The README supplies the benchmark contract and source attribution outside the artwork.

The figure uses **Noto Sans**, with the regular and bold faces packaged in `fonts/`. Their source and checksums are recorded in `fonts/SOURCE.json`; the fonts are distributed under the **SIL Open Font License 1.1** in `fonts/OFL.txt`. The PDF embeds Noto Sans and the PNG preserves its appearance. The SVG retains live Noto Sans text; install the packaged faces when editing the vector artwork. An optional `--font-directory` selects another directory containing these two Noto Sans faces.

The code resolves inputs and fonts relative to its own directory. It registers the packaged Noto Sans faces and checks coverage of every displayed character in both weights. `data.json` records the benchmark sources and values.
