import time
import os
import pandas as pd
from vlm import VLM
from sampling import uniform_sample, random_sample, keyframe_sample
from prompt import prompt

VIDEO_DIR = './data/videos'
DATA_PATH = './data/sampled_data.csv'
RESULTS_DIR = './data/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_SIZES = [24, 18, 12, 6, 3]
ACTION_TYPES = ['Turnover', 'Foul', 'Block', 'Rebound', 'Steal', '2PT Shot', '3PT Shot', 'Free Throw', 'Violation']

def extract_action_from_caption(caption):
    """Extract action type from caption string."""
    for action in ACTION_TYPES:
        if action in str(caption):
            return action
    return 'Unknown'

def run_experiment(vlm, df, method, n_frames=None):
    """Run a single experiment and save results to CSV."""
    if n_frames is not None:
        filename = f"{RESULTS_DIR}/{method}_{n_frames}_results.csv"
    else:
        filename = f"{RESULTS_DIR}/{method}_results.csv"

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
    
    print(f"[{method} n={n_frames}] {len(pending)} clips remaining")

    for i, (counter, row) in enumerate(pending):
        video_path = f"{VIDEO_DIR}/clip_{row['original_index'] + 1}.mp4"
        try:
            start = time.time()
            if method == 'uniform':
                frames = uniform_sample(video_path, n_frames)
            elif method == 'random':
                frames = random_sample(video_path, n_frames)
            elif method == 'keyframe':
                frames = keyframe_sample(video_path)

            caption = vlm.get_response(frames)
            elapsed = time.time() - start

            action = extract_action_from_caption(caption)

            results.append({
                'url': row['urls'],
                'caption': caption,
                'time': elapsed,
                'n_frames_requested': n_frames,
                'n_frames_used': len(frames),
                'action': action
            })

        except Exception as e:
            print(f"Failed on clip {counter + 1}: {e}")
            results.append({
                'url': row['urls'],
                'caption': '',
                'time': None,
                'n_frames_requested': n_frames,
                'n_frames_used': None
            })

        # Save every 5 clips
        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(filename, index=False)
            print(f"[{method} n={n_frames}] Saved progress — {i + 1} clips processed")

    # Final save
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"Completed {filename}")


def main():
    df = pd.read_csv(DATA_PATH, sep=';')
    vlm = VLM(model="unsloth/Qwen2.5-VL-3B-Instruct", system_prompt=prompt)

    # Uniform sampling
    for n in SAMPLE_SIZES:
        run_experiment(vlm, df, 'uniform', n)

    # Random sampling
    for n in SAMPLE_SIZES:
        run_experiment(vlm, df, 'random', n)

    # Keyframe extraction
    run_experiment(vlm, df, 'keyframe')


if __name__ == "__main__":
    main()