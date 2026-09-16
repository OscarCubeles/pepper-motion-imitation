# Scientific IK Evaluation Dashboard

Local, thesis-oriented analysis of the Pepper imitation system's constrained solution,
unconstrained ablation, IKPy baseline, and original/source solution.

Everything in this project is isolated under `04-evaluation/scientific-dashboard/`. The
existing evaluation code and result files are read-only inputs.

## Quick start

Requirements:

- Python 3.10 or newer
- Node.js 22.13 or newer
- pnpm

From this folder:

```powershell
pnpm install
pnpm run refresh
pnpm run dev
```

Open the local address printed by the development server.

## Refresh the analysis

Run one command whenever either source JSON changes:

```powershell
pnpm run refresh
```

It reads:

- `../results/ik_method_metrics.json`
- `../results/performance_metrics.json`

It regenerates:

- `public/data/analysis.json`
- `public/data/video_quality.csv`
- `public/data/video_singularities.csv`
- `public/data/paired_comparisons.csv`
- `public/data/performance_repeats.csv`

The script uses only the Python standard library. Override paths with
`python scripts/build_analysis.py --help` when needed.

## Statistical protocol

- The primary paired unit is the complete video (`n = 25`), never an individual frame.
- Left and right summaries are averaged for arm-specific metrics.
- HJL is counted once because it is a complete two-arm frame indicator duplicated in
  the arm summaries.
- TSE values for frames 0–1 are structurally undefined, not measurement failures.
- WOM valid/missing coverage is retained per method and video.
- Mean paired effects use 10,000 video-level bootstrap resamples.
- Two-sided sign-flip tests use 100,000 permutations.
- Matched-pairs rank-biserial correlations quantify effect size.
- Holm correction is applied across all 21 constrained-versus-baseline comparisons.
- The fixed random seed is `20260729`.
- Category figures are descriptive because each category contains five videos.

## Thesis figures

Every chart offers:

- `SVG` for vector placement in LaTeX, Word, or a vector editor.
- `PNG · 3×` for high-resolution raster placement.
- CSV downloads for reproducing or restyling the active analysis.

The Quality, Category, and Performance sections each have independent controls. By
default they show all methods as vertical video-level mean bars with mean labels.
The Singularity Quality section reuses all seven quality metrics and filters complete
videos by any, dual, elbow-roll, shoulder-roll, or absent singularity diagnostics.
Its classification table reports complete-frame counts and its active CSV export
includes the selected videos' classification flags.
Before exporting, use the native checkboxes to select methods, hide labels, add the
underlying video points, or add an in-figure legend. The paired-effects section has
the same label and legend controls plus selectable baselines; the constrained method
remains its reference. The Frame Explorer has its own method and legend controls.
Exports capture exactly the currently visible figure state.

The optional min-max whisker spans the lowest and highest displayed video summary
for a bar (or the five clips in a motion category); it is not a confidence interval.
The Frame Explorer and Performance evolution figure both support inclusive start/end
frame ranges. Performance evolution uses the mean of the three available repeats and
offers only raw frame-native measurements, not the derived latency-jitter summary.

Use SVG whenever the thesis toolchain supports it. Keep the caption explicit about the
unit of analysis, confidence interval, metric direction, and whether a panel is
descriptive or inferential.

## Validation

```powershell
pnpm test
```

This runs the analysis unit tests, the production build, and a server-rendering check.
