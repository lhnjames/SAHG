"""
data/preprocess_botsim.py
==========================
Preprocess BotSim-24 CSV+JSON → features.csv + labels.csv

BotSim-24 dataset:
  - 2907 users: 1907 human / 1000 bot
  - Source: Reddit-based simulation framework
  - CSV: user profile (name, description, submission/comment counts, subreddit)
  - JSON: per-user post & comment interaction data

Feature extraction strategy:
  - From CSV: numeric activity counts, binary features (has_description, etc.)
  - From JSON: text-based behavioral features per user (n_posts, avg_score, etc.)

Usage:
    python data/preprocess_botsim.py \
        --csv  BotSim-24_Data/BotSim-24_user.csv \
        --json BotSim-24_Data/BotSim-24_user_post_comment.json \
        --out_dir data/botsim_24/
"""

import argparse
import ast
import json
import os
import numpy as np
import pandas as pd


def parse_subreddits(sr_str) -> list:
    """Parse the subreddit field (stored as a string repr of a list)."""
    if pd.isna(sr_str) or not sr_str:
        return []
    try:
        return ast.literal_eval(sr_str)
    except Exception:
        return []


def extract_json_features(uid: str, json_data: dict) -> dict:
    """Extract behavioural features from the user's posts/comments in json."""
    uid = str(uid)
    user_data = json_data.get(uid, {})
    posts     = user_data.get("posts", [])
    comm1     = user_data.get("comment_1", [])
    comm2     = user_data.get("comment_2", [])

    n_posts = len(posts)
    n_comm1 = len(comm1)
    n_comm2 = len(comm2)

    scores    = [p.get("score", 0) or 0 for p in posts]
    n_comms_p = [p.get("num_comments", 0) or 0 for p in posts]
    upvotes   = [p.get("upvote_ratio", 0.5) or 0.5 for p in posts]
    post_lens = [len(p.get("posts", "") or "") for p in posts]

    c1_scores = [c.get("comment_score", 0) or 0 for c in comm1]
    c1_lens   = [len(c.get("comment_body", "") or "") for c in comm1]

    return {
        "n_posts":             n_posts,
        "n_comments_received": int(np.sum(n_comms_p)) if n_comms_p else 0,
        "n_comments_made":     n_comm1 + n_comm2,
        "avg_post_score":      float(np.mean(scores)) if scores else 0.0,
        "avg_upvote_ratio":    float(np.mean(upvotes)) if upvotes else 0.5,
        "avg_post_len":        float(np.mean(post_lens)) if post_lens else 0.0,
        "avg_comment_score":   float(np.mean(c1_scores)) if c1_scores else 0.0,
        "avg_comment_len":     float(np.mean(c1_lens)) if c1_lens else 0.0,
        "n_unique_subs":       len(set(p.get("subreddit", "") for p in posts)),
        "is_active":           int((n_posts + n_comm1 + n_comm2) > 0),
    }


def process_botsim(csv_path: str, json_path: str, out_dir: str):
    """Read BotSim-24 CSV + JSON and produce features.csv + labels.csv."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"Loading JSON: {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    labels_arr = df["class"].values.astype(np.int64)

    # --- CSV-based features ---
    df["has_description"]     = df["description"].notna().astype(int)
    df["description_len"]     = df["description"].fillna("").apply(len)
    df["n_subreddits"]        = df["subreddit"].apply(lambda x: len(parse_subreddits(x)))
    # fill numeric NaN with 0
    for col in ["submission_num", "comment_num", "comment_num_1", "comment_num_2"]:
        df[col] = df[col].fillna(0.0)

    csv_feat_cols = [
        "submission_num", "comment_num", "comment_num_1", "comment_num_2",
        "has_description", "description_len",
        "n_subreddits",
    ]

    # --- JSON-based features ---
    json_rows = [extract_json_features(uid, json_data) for uid in df["user_id"]]
    json_df   = pd.DataFrame(json_rows)

    feat_df = pd.concat([df[csv_feat_cols].reset_index(drop=True),
                         json_df.reset_index(drop=True)], axis=1)

    feat_path  = os.path.join(out_dir, "features.csv")
    label_path = os.path.join(out_dir, "labels.csv")
    feat_df.to_csv(feat_path, index=False)
    pd.DataFrame({"label": labels_arr}).to_csv(label_path, index=False)

    print(f"\nDone! {len(feat_df)} users.")
    print(f"  bot={labels_arr.sum()}, human={(labels_arr==0).sum()}")
    print(f"  Features ({feat_df.shape[1]} dims) saved to {feat_path}")
    print(f"  Labels saved to {label_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True)
    parser.add_argument("--json",    required=True)
    parser.add_argument("--out_dir", default="data/botsim_24")
    args = parser.parse_args()
    process_botsim(args.csv, args.json, args.out_dir)
