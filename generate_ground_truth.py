from vlm import VLM
from prompt import prompt
import time
import os
import pandas as pd

ACTION_TYPES = ['Turnover', 'Foul', 'Block', 'Rebound', 'Steal', '2PT Shot', '3PT Shot', 'Free Throw', 'Violation']

def extract_action_from_caption(caption):
    """Extract action type from caption string."""
    for action in ACTION_TYPES:
        if action in str(caption):
            return action
    return 'Unknown'

def run_experiment(vlm, df):
    """Run a single experiment and save results to CSV."""

    filename = "./data/ground_truth.csv"

    # Load existing results if any
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        completed_urls = set(existing_df['url'].tolist())
        results = existing_df.to_dict('records')
        print(f"Resuming {filename} — {len(completed_urls)} clips already done")
    else:
        completed_urls = set()
        results = []

    pending = [(counter, row) for counter, row in df.iterrows() 
               if row['urls'] not in completed_urls]
    
    print(f"{len(pending)} clips remaining")

    for i, (counter, row) in enumerate(pending):
        video_path = f"./data/videos/clip_{row['original_index'] + 1}.mp4"
        try:
            start = time.time()
            caption, tokens = vlm.get_response_video(video_path, n_frames=100)
            elapsed = time.time() - start
            
            action = extract_action_from_caption(caption)

            results.append({
                'url': row['urls'],
                'caption': caption,
                'time': elapsed,
                'tokens': tokens,
                'action': action
            })

        except Exception as e:
            print(f"Failed on clip {counter + 1}: {e}")
            results.append({
                'url': row['urls'],
                'caption': '',
                'time': None,
                'tokens': None,
                'action': None
            })

        # Save every 5 clips
        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(filename, index=False)
            print(f"Saved progress — {i + 1} clips processed")


    # Final save
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Completed {filename}")

def main():
    df = pd.read_csv('./data/sampled_data.csv', sep=';')
    vlm = VLM(model = "xai.grok-4.20-reasoning", system_prompt = prompt)
    run_experiment(vlm, df)

if __name__ == "__main__":
    main()