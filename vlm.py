import openai
import base64
import cv2

LOCAL_MODELS = {
    "unsloth/Qwen2.5-VL-3B-Instruct": "http://localhost:8085/v1",
    "Qwen/Qwen3-VL-8B-Instruct": "http://localhost:8086/v1",
}

class VLM():
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(
            base_url=LOCAL_MODELS[model],
            api_key="dummy"
        )

    def encode_frame(self, frame) -> str:
        """Encode a cv2 frame to base64 JPEG."""
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def get_frames_full_video(self, video_path: str) -> list:
        """Extract ALL frames from video for ground truth generation."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
        cap.release()
        return frames

    def get_response(self, frames: list) -> str:
        """
        Send frames to the VLM and get a caption back.

        Args:
            frames: list of cv2 frames

        Returns:
            string caption
        """
        content = []
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_frame(frame)}"
                }
            })
        content.append({
            "type": "text",
            "text": self.system_prompt
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=128,
            temperature=0.1
        )

        return response.choices[0].message.content

    def get_response_video(self, video_path: str, n_frames: int) -> str:
        """Generate caption for full video."""
        frames = self.get_frames_full_video(video_path)
        if len(frames) <= n_frames:
            filtered_frames = frames
        else:
            indices = [int(i * len(frames) / n_frames) for i in range(n_frames)]
            filtered_frames = [frames[i] for i in indices]
        return self.get_response(filtered_frames)