import pandas as pd
import os
from pycocoevalcap.cider.cider import Cider

RESULTS_DIR = './data/results'
GROUND_TRUTH_PATH = './data/ground_truth.csv'
SAMPLE_SIZES = [3, 6, 12, 18, 24]
ACTION_TYPES = ['Turnover', 'Foul', 'Block', 'Rebound', 'Steal', '2PT Shot', '3PT Shot', 'Free Throw', 'Violation']

def compute_cider(ground_truth_df, results_df, action=None):
    """Compute CIDEr score between results and ground truth, optionally filtered by action."""
    
    # Merge on url
    merged = results_df.drop(columns=['action'], errors='ignore').merge(
        ground_truth_df[['url', 'caption', 'action']], 
        on='url', 
        suffixes=('_pred', '_gt')
    )
    
    # Filter by action if specified
    if action is not None:
        merged = merged[merged['action'] == action]
    
    if len(merged) == 0:
        return None, 0

    # Build CIDEr input format
    gts = {}
    res = {}
    for i, row in merged.iterrows():
        gts[i] = [row['caption_gt']]
        res[i] = [row['caption_pred']]
    
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score, len(merged)


def evaluate(ground_truth_df, results_df, method, n_frames):
    """Evaluate overall and per-action CIDEr scores."""
    results = []

    # Overall score
    score, n_clips = compute_cider(ground_truth_df, results_df)
    results.append({
        'method': method,
        'n_frames': n_frames,
        'action': 'overall',
        'cider_score': score,
        'n_clips': n_clips
    })

    # Per-action scores
    for action in ACTION_TYPES:
        score, n_clips = compute_cider(ground_truth_df, results_df, action=action)
        if score is not None:
            results.append({
                'method': method,
                'n_frames': n_frames,
                'action': action,
                'cider_score': score,
                'n_clips': n_clips
            })

    return results


def main():
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    all_results = []

    for size in SAMPLE_SIZES:
        results_df = pd.read_csv(RESULTS_DIR + f"/random_{size}_results.csv")
        all_results.extend(evaluate(ground_truth_df, results_df, 'random', size))

    for size in SAMPLE_SIZES:
        results_df = pd.read_csv(RESULTS_DIR + f"/uniform_{size}_results.csv")
        all_results.extend(evaluate(ground_truth_df, results_df, 'uniform', size))

    results_df = pd.read_csv(RESULTS_DIR + "/keyframe_results.csv")
    all_results.extend(evaluate(ground_truth_df, results_df, 'keyframe', None))

    eval_results = pd.DataFrame(all_results).sort_values(['method', 'action', 'n_frames'])
    eval_results.to_csv('./data/cider_results.csv', index=False)
    print(eval_results.to_string())
    print("\nSaved to ./data/cider_results.csv")


if __name__ == "__main__":
    main()