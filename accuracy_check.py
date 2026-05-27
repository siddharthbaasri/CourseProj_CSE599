import pandas as pd

SAMPLED_DATA_PATH = './data/sampled_data.csv'
GROUND_TRUTH_PATH = './data/ground_truth.csv'

def main():
    sampled_df = pd.read_csv(SAMPLED_DATA_PATH, sep=';')
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)

    # Rename for consistency
    sampled_df = sampled_df.rename(columns={'urls': 'url'})

    # Merge on url
    merged = sampled_df[['url', 'main_action']].merge(
        ground_truth_df[['url', 'action']],
        on='url'
    )

    # Compare main_action vs action
    merged['correct'] = merged['main_action'] == merged['action']

    accuracy = merged['correct'].mean() * 100
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Correct: {merged['correct'].sum()} / {len(merged)}")

    # Per action breakdown
    print("\nPer action accuracy:")
    print(merged.groupby('main_action')['correct'].mean().mul(100).round(2))

if __name__ == "__main__":
    main()