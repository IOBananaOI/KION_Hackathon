import os

from utils import *
from get_caps import get_captions
from functools import partial

from multiprocessing import Pool

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

if __name__ == '__main__':
#### Step 1: Get shots from film

    shots = get_shots(VIDEO_PATH, show_progress=True)
    print("Step 1 done.")


    # #### Step 2: Save one image from each shot

    save_images_from_shots(shots, VIDEO_PATH, IMG_PATH)
    print("Step 2 done")


    # #### Step 3: Get start/end shot pairs for each scene

    scenes = get_scenes(IMG_PATH)
    scenes_count = len(scenes)

    print("Step 3 done.")

    # Get scene start/end timecodes

    scenes_intervals = [(shots[interval[0]][0].get_timecode(), shots[interval[1]][0].get_timecode()) for interval in scenes]


    # #### Step 4: Cut film into scenes

    cut_video(VIDEO_PATH, SCENES_PATH, scenes_intervals)
    print("Step 4 done.")


    #### Step 5: Get captions

    captions = get_captions(SCENES_PATH, 11)
    print("Step 5 done.")


    #### Step 6: Convert text captions to voice captions

    txt_cap2voice_cap(['a']*11, VOICE_CAPTION_DIR)
    print("Step 6 done.")


    #### Step 7: Composite original scenes with voice captions

    scenes_names = list(map(lambda x: x[:-4], os.listdir('scenes')))
    t = time()
    with Pool(4) as p: 
        p.map(partial(wrapped_scene_composition, SCENES_PATH, VOICE_CAPTION_DIR, RESULTING_SCENES), scenes_names)

    print("Step 7 done.")

    #### Final step: Concatenate all scenes

    concatenate_scenes(RESULTING_SCENES, VIDEO_PATH, 11)

    print("Final step done.")
    print(f"Working time: {sum(worktime)} seconds.")


    #### Clear unnecessary files

    files_clear()


    #### Export worktime results
    export_worktime(worktime, funcs)