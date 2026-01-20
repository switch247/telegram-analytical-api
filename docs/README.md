# Project Documentation

This directory contains reusable, professional documentation for the project.

## Folder Structure

```
credit-risk-model-week4/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── README.md                # Reusable project docs (this file)
│   ├── business_understanding.md# Task 1 business understanding
│   ├── dependencies.md          # Environment & tooling overview
│   └── notebooks.md             # Notebook workflow guidance
├── experiments/
│   └── todo.md                  # Challenge brief & references
├── notebooks/                   # Exploratory analysis
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── reports/
├── scripts/                     # Utility/CLI scripts
├── src/                         # Library code: data, features, pipeline, utils
├── tests/                       # Unit tests
├── dvc.yaml                     # Data pipeline definition (optional)
├── docker-compose.yml           # Local services (optional)
├── pyproject.toml               # Build & tooling config
├── requirements.txt             # Python dependencies
└── README.md                    # Project-specific overview
```

## How to Run (High Level)

- Set up a Python virtual environment.
- Install project dependencies from `requirements.txt`.
- Prepare or reproduce processed data.
- Run notebooks for EDA and insights.
- Execute scripts or pipelines to train and evaluate models.
- Serve the model via an API if required.

See the main project README for copy-ready commands tailored to this repository.

## Conventions

- Keep project-specific details in the root `README.md`.
- Place reusable guidance and references under `docs/`.
- Prefer interpretable baselines first; add explainability for complex models.
- Record experiments and decisions for traceability.
