from vlm import VLM
from prompt import prompt
import time

def main():
    video_path = './data/videos/clip_1.mp4'
    vlm = VLM(model = "unsloth/Qwen2.5-VL-3B-Instruct", system_prompt = prompt)
    start = time.time()
    response = vlm.get_response_video(video_path)
    print(f"Time: {time.time() - start:.2f}s")
    print(response)

if __name__ == "__main__":
    main()