"""
tools/clustering_tools.py
─────────────────────────────────────────────────────────────────────────────
Three LangChain tools that give the ReAct agent the ability to:
  1. Preprocess the German Credit dataset
  2. Train a mixed-data clustering pipeline (Gower + Agglomerative)
  3. Analyse a brand-new applicant against the learned clusters

Serialisation strategy
─────────────────────────────────────────────────────────────────────────────
  • The full fitted pipeline (encoder + scaler + cluster labels) is saved as
    a joblib artefact: models/clustering_pipeline.joblib
  • The clustered dataset is saved as: models/clustered_customers.csv
  • A cluster-profile summary JSON is saved as: models/cluster_profiles.json

These artefacts are read back on every call to `analyze_new_customer`, so the
agent can run the tools in separate steps without losing state.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import gower
import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn_extra.cluster import KMedoids

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_PIPELINE_PATH = _MODELS_DIR / "clustering_pipeline.joblib"
_CLUSTERED_CSV = _MODELS_DIR / "clustered_customers.csv"
_PROFILES_JSON = _MODELS_DIR / "cluster_profiles.json"

# ─────────────────────────────────────────────────────────────────────────────
# Feature schema
# ─────────────────────────────────────────────────────────────────────────────
NUMERICAL_COLS = ["Age", "Job", "Credit amount", "Duration"]
CATEGORICAL_COLS = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# Ordered levels used for ordinal encoding
ORDINAL_MAPS: Dict[str, list] = {
    "Saving accounts": ["Unknown", "little", "moderate", "quite rich", "rich"],
    "Checking account": ["Unknown", "little", "moderate", "rich"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper – build a human-readable cluster profile string
# ─────────────────────────────────────────────────────────────────────────────
def _build_profile_string(profiles: Dict[str, Any]) -> str:
    lines: list[str] = []
    for cluster_id, stats in profiles.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"  Cluster {cluster_id}  (n={stats['size']})")
        lines.append(f"{'='*60}")
        lines.append(f"  Avg Age           : {stats['avg_age']:.1f} yrs")
        lines.append(f"  Avg Credit Amount : €{stats['avg_credit_amount']:,.0f}")
        lines.append(f"  Avg Loan Duration : {stats['avg_duration']:.1f} months")
        lines.append(f"  Avg Job Level     : {stats['avg_job']:.2f}  (0=unskilled, 3=highly skilled)")
        lines.append(f"  Top Purpose       : {stats['top_purpose']}")
        lines.append(f"  Top Housing       : {stats['top_housing']}")
        lines.append(f"  Top Saving Acct   : {stats['top_saving_accounts']}")
        lines.append(f"  Top Checking Acct : {stats['top_checking_account']}")
        lines.append(f"  Gender Split      : {stats['gender_split']}")
        lines.append(f"  Risk Assessment   : {stats['risk_label']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helper – derive a risk label from cluster statistics
# ─────────────────────────────────────────────────────────────────────────────
def _derive_risk_label(stats: Dict[str, Any]) -> str:
    risk_score = 0

    # High credit amount relative to overall mean → +risk
    if stats["avg_credit_amount"] > 3500:
        risk_score += 2
    elif stats["avg_credit_amount"] > 2000:
        risk_score += 1

    # Long duration → +risk
    if stats["avg_duration"] > 30:
        risk_score += 2
    elif stats["avg_duration"] > 18:
        risk_score += 1

    # Poor saving account → +risk
    if stats["top_saving_accounts"] in ("Unknown", "little"):
        risk_score += 2

    # Poor checking account → +risk
    if stats["top_checking_account"] in ("Unknown", "little"):
        risk_score += 2

    # Young borrowers → +risk
    if stats["avg_age"] < 28:
        risk_score += 1

    if risk_score >= 6:
        return "⚠️  HIGH RISK – thin savings buffer, large/long credit, young borrowers"
    elif risk_score >= 3:
        return "⚡ MEDIUM RISK – moderate financial stability, review case by case"
    else:
        return "✅ LOW RISK – established customers with solid savings and short durations"


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 1 – preprocess_german_credit_data
# ═════════════════════════════════════════════════════════════════════════════
@tool
def preprocess_german_credit_data(file_path: str) -> str:
    """
    Load and preprocess the German Credit dataset from the given CSV file path.

    What this tool does
    ───────────────────
    • Reads the CSV file from `file_path`.
    • Drops the anonymous row-index column ('Unnamed: 0') if present.
    • Fills missing values:
        - 'Saving accounts'  → fills NaN with 'Unknown'
        - 'Checking account' → fills NaN with 'Unknown'
        - Any remaining numerical NaN → median imputation
    • Validates that all required feature columns are present.
    • Saves the cleaned DataFrame to 'models/preprocessed_data.csv'.
    • Returns a plain-text summary of the cleaned dataset (shape, dtypes,
      missing-value counts, and basic descriptive statistics).

    When to use
    ───────────
    Call this tool FIRST, before training or inference. The tool must receive
    a valid local file path string, e.g. 'data/german_credit_data.csv'.

    Args
    ────
    file_path : str
        Relative or absolute path to the raw German Credit CSV file.

    Returns
    ───────
    str  – Human-readable preprocessing report.
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = _ROOT / file_path
        if not path.exists():
            return f"ERROR: File not found at '{path}'. Please check the path."

        df = pd.read_csv(path)

        # Drop anonymous index column
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        # Verify required columns
        missing_cols = [c for c in ALL_FEATURE_COLS if c not in df.columns]
        if missing_cols:
            return f"ERROR: Missing expected columns: {missing_cols}. Found: {list(df.columns)}"

        initial_missing = df.isnull().sum()

        # Impute categorical columns with 'Unknown'
        for col in ["Saving accounts", "Checking account"]:
            df[col] = df[col].fillna("Unknown")

        # Impute any remaining numerical NaN with median
        num_imputer = SimpleImputer(strategy="median")
        df[NUMERICAL_COLS] = num_imputer.fit_transform(df[NUMERICAL_COLS])

        final_missing = df.isnull().sum()

        # Persist cleaned dataset
        cleaned_path = _MODELS_DIR / "preprocessed_data.csv"
        df.to_csv(cleaned_path, index=False)

        # Build report
        report_lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║         PREPROCESSING REPORT – German Credit Data        ║",
            "╚══════════════════════════════════════════════════════════╝",
            f"  Dataset shape       : {df.shape[0]:,} rows × {df.shape[1]} columns",
            "",
            "  Missing values BEFORE imputation:",
        ]
        for col, cnt in initial_missing.items():
            if cnt > 0:
                report_lines.append(f"    • {col:<22}: {cnt:>4} ({cnt/len(df)*100:.1f}%)")

        report_lines.append("")
        report_lines.append("  Missing values AFTER imputation:")
        total_after = final_missing.sum()
        report_lines.append(
            f"    • Total remaining  : {total_after} ✅" if total_after == 0
            else f"    • Total remaining  : {total_after} ⚠️"
        )

        report_lines += [
            "",
            "  Numerical statistics:",
            df[NUMERICAL_COLS].describe().round(2).to_string(index=True),
            "",
            "  Categorical column value counts:",
        ]
        for col in CATEGORICAL_COLS:
            counts = df[col].value_counts().to_dict()
            report_lines.append(f"    {col}: {counts}")

        report_lines.append(f"\n  ✅ Cleaned data saved to: {cleaned_path}")

        return "\n".join(report_lines)

    except Exception as exc:  # noqa: BLE001
        return f"PREPROCESSING ERROR: {exc}"


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 2 – train_mixed_data_clustering
# ═════════════════════════════════════════════════════════════════════════════
@tool
def train_mixed_data_clustering(n_clusters: int = 4) -> str:
    """
    Train a mixed-data clustering pipeline on the preprocessed German Credit
    dataset and return a detailed cluster-profile report.

    Algorithm choice
    ────────────────
    Standard k-means cannot handle categorical features.  This tool uses a
    two-stage approach that is well-suited to mixed data:

      Stage 1 – Representation
        • Numerical features are scaled to [0, 1] with MinMaxScaler.
        • Categorical features with known ordinal semantics ('Saving accounts',
          'Checking account') are label-encoded into integers reflecting their
          natural order (Unknown < little < moderate < quite rich < rich).
        • All remaining categorical columns are one-hot encoded.
        • A Gower Distance matrix is computed across all features.  Gower
          distance natively handles mixed types (Manhattan distance for numbers,
          Dice distance for binary/categorical indicators) and outputs a
          symmetric n×n dissimilarity matrix.

      Stage 2 – Clustering
        • K-Medoids clustering is applied to the Gower distance matrix.
          K-Medoids (PAM) uses actual data-points as cluster centres, making
          it far more interpretable and robust to outliers than k-means.

    What this tool saves
    ────────────────────
    • models/clustering_pipeline.joblib  – fitted objects (encoders, scaler,
                                           medoid indices, cluster labels)
    • models/clustered_customers.csv     – original data + 'Cluster' column
    • models/cluster_profiles.json       – per-cluster statistics

    When to use
    ───────────
    Call this tool AFTER preprocessing.  It expects
    'models/preprocessed_data.csv' to exist.  You may pass an integer
    `n_clusters` (default=4).

    Args
    ────
    n_clusters : int
        Number of customer segments to discover (default 4, valid range 2–8).

    Returns
    ───────
    str  – A rich cluster-profile report describing each segment.
    """
    try:
        cleaned_path = _MODELS_DIR / "preprocessed_data.csv"
        if not cleaned_path.exists():
            return (
                "ERROR: Preprocessed data not found at 'models/preprocessed_data.csv'. "
                "Please run 'preprocess_german_credit_data' first."
            )

        df = pd.read_csv(cleaned_path)

        if n_clusters < 2 or n_clusters > 8:
            return "ERROR: n_clusters must be between 2 and 8."

        # ── 1. Feature engineering ──────────────────────────────────────────

        work = df[ALL_FEATURE_COLS].copy()

        # Ordinal encoding for account columns
        ordinal_encoders: Dict[str, Dict[str, int]] = {}
        for col, levels in ORDINAL_MAPS.items():
            mapping = {lvl: i for i, lvl in enumerate(levels)}
            work[col] = work[col].map(mapping).fillna(0).astype(int)
            ordinal_encoders[col] = mapping

        # Nominal one-hot encoding for remaining categoricals
        nominal_cols = [c for c in CATEGORICAL_COLS if c not in ORDINAL_MAPS]
        work = pd.get_dummies(work, columns=nominal_cols, drop_first=False)

        # MinMax scale ALL columns (Gower expects values in [0,1] for num cols)
        scaler = MinMaxScaler()
        work_scaled = pd.DataFrame(
            scaler.fit_transform(work),
            columns=work.columns,
        )

        # Boolean mask: True = column is categorical (binary after OHE)
        is_cat_mask = np.array(
            [False] * len(NUMERICAL_COLS) +
            [True] * (work_scaled.shape[1] - len(NUMERICAL_COLS))
        )

        # ── 2. Gower distance ───────────────────────────────────────────────
        X_np = work_scaled.values.astype(float)
        gower_mat = gower.gower_matrix(X_np, cat_features=is_cat_mask)

        # ── 3. K-Medoids ────────────────────────────────────────────────────
        kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            method="pam",
            init="k-medoids++",
            random_state=42,
            max_iter=500,
        )
        kmedoids.fit(gower_mat)
        labels = kmedoids.labels_

        # ── 4. Attach labels & profile ──────────────────────────────────────
        df_out = df.copy()
        df_out["Cluster"] = labels

        profiles: Dict[str, Any] = {}
        for cid in sorted(df_out["Cluster"].unique()):
            sub = df_out[df_out["Cluster"] == cid]
            stats: Dict[str, Any] = {
                "size": int(len(sub)),
                "avg_age": float(sub["Age"].mean()),
                "avg_credit_amount": float(sub["Credit amount"].mean()),
                "avg_duration": float(sub["Duration"].mean()),
                "avg_job": float(sub["Job"].mean()),
                "top_purpose": str(sub["Purpose"].mode().iloc[0]),
                "top_housing": str(sub["Housing"].mode().iloc[0]),
                "top_saving_accounts": str(sub["Saving accounts"].mode().iloc[0]),
                "top_checking_account": str(sub["Checking account"].mode().iloc[0]),
                "gender_split": sub["Sex"].value_counts().to_dict(),
            }
            stats["risk_label"] = _derive_risk_label(stats)
            profiles[str(cid)] = stats

        # ── 5. Persist artefacts ────────────────────────────────────────────
        pipeline_data = {
            "ordinal_encoders": ordinal_encoders,
            "scaler": scaler,
            "scaler_columns": list(work.columns),
            "is_cat_mask": is_cat_mask,
            "gower_mat": gower_mat,
            "kmedoids": kmedoids,
            "n_clusters": n_clusters,
            "ohe_columns": list(work_scaled.columns),
            "nominal_cols": nominal_cols,
        }
        joblib.dump(pipeline_data, _PIPELINE_PATH)
        df_out.to_csv(_CLUSTERED_CSV, index=False)

        with open(_PROFILES_JSON, "w") as fh:
            # gender_split dict is not JSON-serialisable by default → convert
            json_safe = {}
            for cid, s in profiles.items():
                s2 = dict(s)
                s2["gender_split"] = {str(k): int(v) for k, v in s2["gender_split"].items()}
                json_safe[cid] = s2
            json.dump(json_safe, fh, indent=2)

        report = _build_profile_string(profiles)
        header = (
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║       CLUSTERING TRAINING REPORT – K-Medoids / Gower    ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
            f"  Algorithm : K-Medoids (PAM) + Gower Distance\n"
            f"  Clusters  : {n_clusters}\n"
            f"  Samples   : {len(df):,}\n"
        )
        footer = (
            f"\n  ✅ Pipeline saved   : {_PIPELINE_PATH}\n"
            f"  ✅ Clustered CSV    : {_CLUSTERED_CSV}\n"
            f"  ✅ Profiles JSON    : {_PROFILES_JSON}\n"
        )
        return header + report + footer

    except Exception as exc:  # noqa: BLE001
        import traceback
        return f"CLUSTERING ERROR: {exc}\n{traceback.format_exc()}"


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 3 – analyze_new_customer
# ═════════════════════════════════════════════════════════════════════════════
@tool
def analyze_new_customer(customer_json: str) -> str:
    """
    Classify a new credit applicant into a learned customer cluster and
    produce a full analytical risk report.

    How it works
    ────────────
    1. Loads the fitted pipeline from 'models/clustering_pipeline.joblib'.
    2. Applies the same preprocessing steps (ordinal encoding, OHE, scaling).
    3. Computes the Gower distance from the new customer to every row in the
       training set.
    4. Finds the nearest training data-point and inherits its cluster label.
    5. Loads cluster profiles from 'models/cluster_profiles.json'.
    6. Returns a rich report stating: cluster ID, cluster characteristics,
       risk level, and a plain-language analyst recommendation.

    When to use
    ───────────
    Call this tool AFTER 'train_mixed_data_clustering' has run successfully.
    It expects the pipeline artefacts to exist in the 'models/' directory.

    Args
    ────
    customer_json : str
        A JSON-formatted string (or Python-dict-like string) containing the
        new customer's features.  Required keys:
            Age (int), Sex (str), Job (int), Housing (str),
            Saving accounts (str), Checking account (str),
            Credit amount (int), Duration (int), Purpose (str)

        Example:
            '{\"Age\": 24, \"Sex\": \"male\", \"Job\": 2,
              \"Housing\": \"rent\", \"Saving accounts\": \"little\",
              \"Checking account\": \"moderate\", \"Credit amount\": 4500,
              \"Duration\": 36, \"Purpose\": \"business\"}'

    Returns
    ───────
    str  – Detailed analyst report with cluster assignment and risk opinion.
    """
    try:
        # ── Parse input ─────────────────────────────────────────────────────
        # Be lenient: accept single-quoted dicts from the LLM
        raw = customer_json.strip()
        # Replace Python-style single quotes with double quotes for JSON
        raw_json = re.sub(r"(?<![\\])'", '"', raw)
        try:
            customer: Dict[str, Any] = json.loads(raw_json)
        except json.JSONDecodeError:
            # Last resort: eval (safe here, only called by the agent)
            import ast
            customer = ast.literal_eval(raw)

        # ── Load pipeline ───────────────────────────────────────────────────
        if not _PIPELINE_PATH.exists():
            return (
                "ERROR: Clustering pipeline not found. "
                "Please run 'train_mixed_data_clustering' first."
            )

        pipeline_data = joblib.load(_PIPELINE_PATH)
        scaler: MinMaxScaler = pipeline_data["scaler"]
        ordinal_encoders: Dict[str, Dict[str, int]] = pipeline_data["ordinal_encoders"]
        is_cat_mask: np.ndarray = pipeline_data["is_cat_mask"]
        gower_mat: np.ndarray = pipeline_data["gower_mat"]
        kmedoids: KMedoids = pipeline_data["kmedoids"]
        ohe_columns: list = pipeline_data["ohe_columns"]
        nominal_cols: list = pipeline_data["nominal_cols"]

        # ── Load training features ──────────────────────────────────────────
        if not _CLUSTERED_CSV.exists():
            return "ERROR: Clustered dataset not found. Re-run the training tool."

        df_train = pd.read_csv(_CLUSTERED_CSV)

        # ── Build new-customer row ──────────────────────────────────────────
        # Fill missing keys with defaults
        defaults = {
            "Saving accounts": "Unknown",
            "Checking account": "Unknown",
        }
        for k, v in defaults.items():
            if k not in customer or customer[k] is None or str(customer[k]) == "nan":
                customer[k] = v

        row = pd.DataFrame([customer])
        for col in ALL_FEATURE_COLS:
            if col not in row.columns:
                return f"ERROR: Missing feature '{col}' in customer data."

        work = row[ALL_FEATURE_COLS].copy()

        # Ordinal encode
        for col, mapping in ordinal_encoders.items():
            work[col] = work[col].map(mapping).fillna(0).astype(int)

        # One-hot encode nominals (align to training columns)
        work = pd.get_dummies(work, columns=nominal_cols, drop_first=False)

        # Align columns to training schema
        for col in ohe_columns:
            if col not in work.columns:
                work[col] = 0
        work = work[ohe_columns]

        # Scale
        work_scaled = pd.DataFrame(
            scaler.transform(work),
            columns=ohe_columns,
        )

        # ── Gower distance to every training point ──────────────────────────
        X_new = work_scaled.values.astype(float)

        # Build training matrix the same way
        train_work = df_train[ALL_FEATURE_COLS].copy()
        for col, mapping in ordinal_encoders.items():
            train_work[col] = train_work[col].map(mapping).fillna(0).astype(int)
        train_work = pd.get_dummies(train_work, columns=nominal_cols, drop_first=False)
        for col in ohe_columns:
            if col not in train_work.columns:
                train_work[col] = 0
        train_work = train_work[ohe_columns]
        train_scaled = pd.DataFrame(
            scaler.transform(train_work),
            columns=ohe_columns,
        )

        X_train = train_scaled.values.astype(float)
        combined = np.vstack([X_train, X_new])
        gower_full = gower.gower_matrix(combined, cat_features=is_cat_mask)

        # Distance from new customer (last row) to all training points
        dist_to_train = gower_full[-1, :-1]
        nearest_idx = int(np.argmin(dist_to_train))
        assigned_cluster = int(df_train.iloc[nearest_idx]["Cluster"])

        # ── Load profiles ───────────────────────────────────────────────────
        if not _PROFILES_JSON.exists():
            return "ERROR: Cluster profiles JSON not found. Re-run training."

        with open(_PROFILES_JSON) as fh:
            profiles = json.load(fh)

        cluster_stats = profiles[str(assigned_cluster)]

        # ── Build analyst report ────────────────────────────────────────────
        report_lines = [
            "\n╔══════════════════════════════════════════════════════════╗",
            "║          NEW CUSTOMER ANALYSIS – CLUSTER ASSIGNMENT      ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            "  ── Applicant Profile ──────────────────────────────────",
        ]
        for k, v in customer.items():
            report_lines.append(f"    {k:<22}: {v}")

        report_lines += [
            "",
            f"  ── Cluster Assignment ──────────────────────────────────",
            f"    Assigned Cluster ID  : {assigned_cluster}",
            f"    Cluster Size         : {cluster_stats['size']} historical customers",
            f"    Nearest Neighbour Δ  : {dist_to_train[nearest_idx]:.4f} (Gower distance)",
            "",
            "  ── Cluster Historical Characteristics ─────────────────",
            f"    Avg Age              : {cluster_stats['avg_age']:.1f} yrs",
            f"    Avg Credit Amount    : €{cluster_stats['avg_credit_amount']:,.0f}",
            f"    Avg Loan Duration    : {cluster_stats['avg_duration']:.1f} months",
            f"    Typical Purpose      : {cluster_stats['top_purpose']}",
            f"    Typical Housing      : {cluster_stats['top_housing']}",
            f"    Savings Profile      : {cluster_stats['top_saving_accounts']}",
            f"    Checking Profile     : {cluster_stats['top_checking_account']}",
            "",
            "  ── Risk Assessment ────────────────────────────────────",
            f"    {cluster_stats['risk_label']}",
            "",
            "  ── Analyst Recommendation ─────────────────────────────",
        ]

        # Contextualise the new customer vs. cluster norms
        age_diff = customer.get("Age", 0) - cluster_stats["avg_age"]
        credit_diff = customer.get("Credit amount", 0) - cluster_stats["avg_credit_amount"]
        duration_diff = customer.get("Duration", 0) - cluster_stats["avg_duration"]

        flags: list[str] = []
        if credit_diff > 1500:
            flags.append(
                f"    ⚠️  Credit amount €{customer.get('Credit amount',0):,} is "
                f"€{credit_diff:,.0f} ABOVE cluster average → elevated exposure."
            )
        if duration_diff > 12:
            flags.append(
                f"    ⚠️  Loan duration {customer.get('Duration',0)} months is "
                f"{duration_diff:.0f} months LONGER than cluster average → higher default window."
            )
        if age_diff < -5:
            flags.append(
                f"    ℹ️  Applicant is {abs(age_diff):.0f} years YOUNGER than typical cluster "
                f"member → limited credit history likely."
            )
        if not flags:
            flags.append(
                "    ✅ Applicant's parameters are broadly in line with their cluster's norms."
            )

        report_lines += flags
        report_lines.append("")
        report_lines.append("  ── Final Opinion ──────────────────────────────────────")

        risk_text = cluster_stats["risk_label"]
        if "HIGH" in risk_text:
            opinion = (
                "    CAUTION ADVISED.  This applicant maps to a high-risk cluster.  "
                "Standard underwriting controls should be applied; consider requesting "
                "additional collateral or guarantors before approval."
            )
        elif "MEDIUM" in risk_text:
            opinion = (
                "    MODERATE CAUTION.  This applicant belongs to a mixed-risk cluster.  "
                "Case-by-case review is recommended.  Standard income and employment "
                "verification should suffice for a final decision."
            )
        else:
            opinion = (
                "    LOW RISK PROFILE.  This applicant shares characteristics with a "
                "financially stable segment.  Routine approval process can proceed, "
                "subject to standard credit-check confirmation."
            )

        report_lines.append(opinion)
        report_lines.append("")

        return "\n".join(report_lines)

    except Exception as exc:  # noqa: BLE001
        import traceback
        return f"ANALYSIS ERROR: {exc}\n{traceback.format_exc()}"
