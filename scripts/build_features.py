import os
import sys
from pathlib import Path
import argparse
import pandas as pd

# Ensure project root is on sys.path so `src` imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.xente_features import build_feature_pipeline


def main():
    parser = argparse.ArgumentParser(description='Build model-ready features from Xente raw data')
    parser.add_argument('--input', default='data/raw/data.csv', help='Path to raw Xente CSV (project-root relative)')
    parser.add_argument('--output', default='data/processed/xente_processed.csv', help='Path to save processed CSV (project-root relative)')
    args = parser.parse_args()

    in_path = (ROOT / args.input).resolve()
    out_path = (ROOT / args.output).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    pipe, feat_names = build_feature_pipeline()
    X = pipe.fit_transform(df)

    # Persist as a wide CSV with estimated feature name stubs for traceability
    os.makedirs(out_path.parent, exist_ok=True)
    out_df = pd.DataFrame(X)
    out_df.to_csv(out_path, index=False)
    print(f"Saved processed features to {out_path}")


if __name__ == '__main__':
    main()
