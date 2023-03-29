from gtts import gTTS
from time import time

from utils import *
from get_caps import get_captions

# Path to video
VIDEO_PATH = "videos/Test.mp4"

# Directory, where we save images from shots
IMG_PATH = "images/"

# Directory, where we save scenes
SCENES_PATH = "scenes/" 

# Directory with voice captions
VOICE_CAPTION_DIR = "voice_captions/"

# Directory with captioned scenes
RESULTING_SCENES = "results/"


t = time()


#### Step 1: Get shots from film

shots = get_shots(VIDEO_PATH, show_progress=True)
print("Step 1 done.")


#### Step 2: Save one image from each shot

save_images_from_shots(shots, VIDEO_PATH, IMG_PATH)
print("Step 2 done")


#### Step 3: Get start/end shot pairs for each scene

scenes = get_scenes(IMG_PATH)
scenes_count = len(scenes)

print("Step 3 done.")

# Get scene start/end timecodes

scenes_intervals = [(shots[interval[0]][0].get_timecode(), shots[interval[1]][0].get_timecode()) for interval in scenes]


#### Step 4: Cut film into scenes

cut_video(VIDEO_PATH, SCENES_PATH, scenes_intervals)
print("Step 4 done.")


#### Step 5: Get captions


captions = get_captions(SCENES_PATH, scenes_count)
print("Step 5 done.")
print(captions)


#### Step 6: Convert text captions to voice captions

for i, caption in enumerate(captions):
    myobj = gTTS(text=caption, lang='en', slow=False) 
    myobj.save(VOICE_CAPTION_DIR+f"Scene_cap_{i+1}.mp3") 

print("Step 6 done.")



#### Step 7: Composite original scenes with voice captions

scenes_composite(SCENES_PATH, VOICE_CAPTION_DIR, RESULTING_SCENES, scenes_count)
print("Step 7 done.")

#### Final step: Concatenate all scenes

concatenate_scenes(RESULTING_SCENES, VIDEO_PATH, scenes_count)

print("Final step done.")
print(f"Working time: {int(time()-t)} seconds.")