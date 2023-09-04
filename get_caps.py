import warnings
warnings.filterwarnings('ignore')

import torch

import av
import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from utils import benchmark

@benchmark
def get_captions(scenes_path: str, scenes_count: int) -> list:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model import 
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    captions_list = []

    for i in range(scenes_count):
        # load scene
        video_path = f"{scenes_path}Scene{i+1}.mp4"
        container = av.open(video_path)

        # extract evenly spaced frames from scene
        seg_len = container.streams.video[0].frames
        clip_len = model.config.encoder.num_frames
        indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))

        # generate caption
        gen_kwargs = {
            "min_length": 10, 
            "max_length": 30, 
            "num_beams": 8,
        }
        
        pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
        tokens = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        
        captions_list.append(caption)


    return captions_list

    
