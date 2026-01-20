# Notebooks Guide

Use notebooks for exploration and reporting only; keep production code in `src/`.

## Typical Flow

1. Load and profile raw/processed data.
2. Explore distributions, missingness, and correlations.
3. Engineer and validate candidate features.
4. Prototype models and capture insights.

## Practices

- Keep notebooks deterministic; set random seeds where relevant.
- Save generated figures and tables under `outputs/`.
- Avoid side effects in notebooks; move reusable code to `src/` and import.

## Running

- Open notebooks in your preferred environment.
- Ensure the virtual environment is active and dependencies installed.
- Execute cells top-to-bottom and commit meaningful results.

Refer to the main README for project-specific commands.
