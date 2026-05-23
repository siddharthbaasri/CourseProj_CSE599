import cv2
import random
from scenedetect import detect, ContentDetector

def uniform_sample(video_path: str, n_frames: int) -> list:
    """Sample n_frames uniformly from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    indices = [int(i * total_frames / n_frames) for i in range(n_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def random_sample(video_path: str, n_frames: int, seed: int = 42) -> list:
    """Sample n_frames randomly from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    random.seed(seed)
    indices = sorted(random.sample(range(total_frames), min(n_frames, total_frames)))
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def keyframe_sample(video_path: str) -> list:
    """Sample keyframes based on scene detection."""
    scenes = detect(video_path, ContentDetector())
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    for scene in scenes:
        frame_num = scene[0].get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    return frames