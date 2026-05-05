"""
human_risk_clustering.py  —  K-means risk-attitude clustering of human participants
====================================================================================
Clusters human participants into three behavioural types (Cautious / Neutral /
Aggressive) based on per-subject mean risk-decision scores across the three tasks.

Input  (data/human_data/human_features.csv):
    Subject_ID                  — participant identifier
    CTD_ESI_mean                — mean ESI rating across CTD trials (1–5)
    FIP_alloc_continuous_mean   — mean continuous allocation score across FIP
                                  trials (1–5); computed as (L×1 + M×3 + H×5)/total
    DNC_SI_scaled_mean          — mean DNC strategy index scaled to [1, 5] across
                                  DNC trials; raw SI = logit(VCR/(VCR+HPR+ε)) on
                                  drift steps (nudge≠0), then mapped to [1, 5] via
                                  two-tier percentile method (floor trials ->5;
                                  non-floor mapped by Q1/Q2/Q3 ->4/3/2/1)

Outputs:
    data/human_data/Human_Clustered_Data.csv
    figures/human_clustering/Cluster_Mean_Barplot.png

Feature pre-extraction note:
    The three input columns represent task-specific intermediate metrics rather
    than the final integer R_D used in OLR/AUC steps:
      • CTD_ESI_mean:               ESI rating (same as R_D for CTD)
      • FIP_alloc_continuous_mean:  continuous weighted average before K-means
                                    discretisation
      • DNC_SI_scaled_mean:         logit-form strategy index scaled to [1,5]
                                    before K-means discretisation
    Pre-extraction code is in experiments/HumanExperimentConfig/ (browser
    interface) or from the raw Google Sheets export — see data/human_data/README.md.

Usage (from repository root):
    python analysis/5_human_clustering/human_risk_clustering.py

    # To use the included sample file instead of the full dataset:
    python analysis/5_human_clustering/human_risk_clustering.py --input data/human_data/test.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).parent.parent.parent          # NeurIPS_CodeSharing/
FIG_DIR  = ROOT / 'figures' / 'human_clustering'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (blue / teal / red)
C_COLORS = {0: '#0077BB', 1: '#009988', 2: '#CC3311'}
C_NAMES  = {0: 'Cautious', 1: 'Neutral', 2: 'Aggressive'}

FEATURES = [
    'CTD_ESI_mean',
    'FIP_alloc_continuous_mean',
    'DNC_SI_scaled_mean',
]
FEAT_LABELS = {
    'CTD_ESI_mean':              'CTD  (ESI, 1–5)',
    'FIP_alloc_continuous_mean': 'FIP  (alloc score, 1–5)',
    'DNC_SI_scaled_mean':        'DNC  (SI scaled, 1–5)',
}


# ─────────────────────────────────────────────────────────────────────────────
def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f'[ERROR] Feature file not found: {path}\n'
            f'  Download the full dataset from HuggingFace, or run with\n'
            f'  --input data/human_data/test.csv for the included sample.'
        )
    df = pd.read_csv(path)
    missing = [c for c in ['Subject_ID'] + FEATURES if c not in df.columns]
    if missing:
        sys.exit(f'[ERROR] Missing columns in {path.name}: {missing}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
def cluster(df: pd.DataFrame) -> np.ndarray:
    """
    StandardScaler + KMeans(k=3, seed=42).
    Cluster labels are re-ordered so that the cluster with the lowest overall
    mean feature value receives label 0 (Cautious) and the highest receives
    label 2 (Aggressive).
    """
    X        = df[FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans   = KMeans(n_clusters=3, random_state=42, n_init=20)
    raw_lbl  = kmeans.fit_predict(X_scaled)

    # Sort by ascending overall mean (not per-feature) so Cautious=0 globally
    cmeans = [X[raw_lbl == i].mean() for i in range(3)]
    order  = np.argsort(cmeans)
    lmap   = {order[k]: k for k in range(3)}
    return np.array([lmap[l] for l in raw_lbl])


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, labels: np.ndarray) -> None:
    print('\n=== K-means clustering (k=3) ===')
    for cid in range(3):
        mask = labels == cid
        sids = df.loc[mask, 'Subject_ID'].values
        print(f'\n  {C_NAMES[cid]}  (n={mask.sum()}):')
        for feat in FEATURES:
            vals = df.loc[mask, feat].values
            print(f'    {feat}: mean={vals.mean():.2f}  SD={vals.std():.2f}')


# ─────────────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, labels: np.ndarray, out_path: Path) -> None:
    df_out = df[['Subject_ID']].copy()
    df_out['Cluster']      = labels
    df_out['Cluster_Name'] = [C_NAMES[l] for l in labels]
    df_out.to_csv(out_path, index=False)
    print(f'\nCluster labels saved -> {out_path.relative_to(ROOT)}')


# ─────────────────────────────────────────────────────────────────────────────
def plot_bar_chart(df: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Diagnostic bar chart: mean score per cluster per task.
    Within each panel the cluster with the lowest within-task mean is labelled
    Cautious (blue), the highest Aggressive (red).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Mean Feature Score per Risk-Attitude Cluster',
                 fontsize=13, fontweight='bold')

    for col_idx, feat in enumerate(FEATURES):
        ax = axes[col_idx]

        # Per-panel ordering by within-task mean
        feat_means = {cid: df.loc[labels == cid, feat].values.mean()
                      for cid in range(3)}
        sorted_cids = sorted(feat_means, key=lambda c: feat_means[c])

        bar_means  = [feat_means[sorted_cids[v]] for v in range(3)]
        bar_ses    = [
            df.loc[labels == sorted_cids[v], feat].values.std() /
            np.sqrt((labels == sorted_cids[v]).sum())
            for v in range(3)
        ]
        bar_colors = [C_COLORS[v] for v in range(3)]
        bar_labels = [C_NAMES[v] for v in range(3)]

        bars = ax.bar(
            [0, 1, 2], bar_means, yerr=bar_ses,
            color=bar_colors, width=0.55, alpha=0.85,
            capsize=5, edgecolor='black', linewidth=0.8, zorder=3
        )
        for bar_obj, m in zip(bars, bar_means):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2, m + 0.08,
                f'{m:.2f}', ha='center', va='bottom', fontsize=9
            )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(bar_labels, fontsize=11)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel('Mean Score (1–5)', fontsize=10)
        ax.set_title(FEAT_LABELS[feat], fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(3, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    out_png = FIG_DIR / 'Cluster_Mean_Barplot.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved -> {out_png.relative_to(ROOT)}')


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='Human risk-attitude clustering')
    parser.add_argument(
        '--input', type=Path,
        default=ROOT / 'data' / 'human_data' / 'test.csv',
        help='Path to per-subject feature CSV (default: data/human_data/test.csv). '
             'For the full dataset, download human_features.csv from HuggingFace.',
    )
    parser.add_argument(
        '--out', type=Path,
        default=ROOT / 'data' / 'human_data' / 'Human_Clustered_Data.csv',
        help='Output CSV path for cluster assignments',
    )
    args       = parser.parse_args()
    input_path = args.input.resolve()
    out_path   = args.out.resolve()

    try:
        label = input_path.relative_to(ROOT)
    except ValueError:
        label = input_path

    print(f'Loading features from: {label}')
    df     = load_features(input_path)
    print(f'  {len(df)} participants loaded.')

    labels = cluster(df)
    print_summary(df, labels)
    save_csv(df, labels, out_path)
    plot_bar_chart(df, labels)

    print('\nDone.')


if __name__ == '__main__':
    main()
