import pandas as pd
import os
from pycocoevalcap.cider.cider import Cider

RESULTS_DIR = './data/results'
GROUND_TRUTH_PATH = './data/ground_truth.csv'
SAMPLE_SIZES = [3, 6, 12, 18, 24]

def compute_cider(ground_truth_df, results_df):
    """Compute CIDEr score between results and ground truth."""
    
    # Merge on url
    merged = results_df.merge(ground_truth_df[['url', 'caption']], on='url', suffixes=('_pred', '_gt'))
    
    # Build CIDEr input format
    # {image_id: [caption]} for both prediction and ground truth
    gts = {}
    res = {}
    for i, row in merged.iterrows():
        gts[i] = [row['caption_gt']]
        res[i] = [row['caption_pred']]
    
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return score, len(merged)

def main():
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    results = []
    for size in SAMPLE_SIZES:
        results_df = pd.read_csv(RESULTS_DIR + f"/random_{size}_results.csv")
        score, num_merged = compute_cider(ground_truth_df, results_df)
        result = {
            'method': "random",
            'n_frames': size,
            'cider_score': score,
            'n_clips': num_merged
        }
        results.append(result)
        print(result)
    
    for size in SAMPLE_SIZES:
        results_df = pd.read_csv(RESULTS_DIR + f"/uniform_{size}_results.csv")
        score, num_merged = compute_cider(ground_truth_df, results_df)
        result = {
            'method': "uniform",
            'n_frames': size,
            'cider_score': score,
            'n_clips': num_merged
        }
        results.append(result)
        print(result)
    
    
    results_df = pd.read_csv(RESULTS_DIR + f"/keyframe_results.csv")
    score, num_merged = compute_cider(ground_truth_df, results_df)
    result = {
            'method': "keyframe",
            'n_frames': None,
            'cider_score': score,
            'n_clips': num_merged
    }
    results.append(result)
    print(result)

    eval_results = pd.DataFrame(results).sort_values(['method', 'n_frames'])
    eval_results.to_csv('./data/cider_results.csv', index=False)
    print("\nSaved to ./data/cider_results.csv")


if __name__ == "__main__":
    main()