# MMALS Goal-Adaptive Geo/RL/Forward-Backward Package

This repository-style package contains a reproducible smoke experiment for a goal-adaptive MMALS extension.

<p align="center">
  <a href="./paper/MMALS_Goal_Adaptive_Geo_RL_FB_article.pdf">
    <img src="https://img.shields.io/badge/Open-Article-0B5FFF?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Open PDF">
  </a>
</p>

## What this package demonstrates

The package tests whether the same synthetic MMALS/RC2O-style audit states induce different routing policies under five goals:

1. Maximize accuracy
2. Maximize retention
3. Minimize cost
4. Maximize stability under drift
5. Maximize host specialization

The core result is goal-conditioned route differentiation: different objectives induce different dominant routes.

## Important caveat

This is a smoke/synthetic trace package. It validates instrumentation and does not replace real RC2O-8D training/evaluation evidence.

## Folder structure

```text
paper/        LaTeX source, compiled PDF, and paper figures
data/         CSV outputs from the goal-adaptive smoke run
notebooks/    Colab notebook
reports/      Markdown smoke report
outputs/      Convenience copy of figures and CSVs
scripts/      Notebook generation helper when available
```

## Main files

- `paper/main.tex` - LaTeX article source
- `paper/MMALS_Goal_Adaptive_Geo_RL_FB_article.pdf` - compiled article
- `notebooks/MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_Colab.ipynb` - Colab notebook
- `data/goal_summary.csv` - global goal summary table
- `reports/MMALS_RC2O_8D_Goal_Adaptive_report.md` - smoke report

## Rebuild paper

From the `paper/` folder:

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Reviewer-safe claim

The strongest safe claim is:

> The goal-adaptive smoke run demonstrates that the proposed Geo/RL/Forward-Backward extension does not merely reproduce the RC2O selector. Under identical synthetic audit states, changing the objective vector induces distinct routing policies.

## Next step

Connect this controller read-only to real MMALS RC2O-8D candidate traces. The backbone should remain frozen first, and the controller should be evaluated post-hoc before any online/end-to-end integration.
