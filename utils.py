from typing import Tuple, List
from os import listdir, unlink
from os.path import isfile, join

from scenedetect import open_video, ContentDetector, SceneManager
from scenedetect.scene_manager import save_images
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, afx

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform

import torch
from torchvision.models import resnet101
from torch import nn

from PIL import Image

from gtts import gTTS

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from time import time


global worktime
global funcs

worktime = [0] * 8
funcs = [''] * 8


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()

        worktime[benchmark.counter] = end - start
        funcs[benchmark.counter] = func.__name__
        benchmark.counter += 1


        return result

    return wrapper

benchmark.counter = 0

@benchmark
def get_shots(video_path, threshold=27.0, show_progress=False):
    """Returns a list of start/end timecode and frames pairs for each shot that was found. """

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video, show_progress=show_progress)

    shots = scene_manager.get_scene_list()
    return shots
    
@benchmark
def save_images_from_shots(shots, video_path, img_out_path, show_progress=False):
    video = open_video(video_path)
    save_images(shots, video, num_images=1, output_dir=img_out_path, show_progress=show_progress)


def estimate_scenes_count(distance_matrix: np.ndarray) -> int:
    """
    Calculate approximate count of scenes.
    Get singular values of the distance_matrix and then - index of the "elbow value".
    :paran distance_matrix: matrix of the pairvaise distances between shots
    :return: estimated count of scenes
    """
    singular_values = np.linalg.svd(distance_matrix, full_matrices=False, compute_uv=False)
    singular_values = singular_values[:len(singular_values) // 2]
    singular_values = np.log(singular_values)

    start_point = np.array([0, singular_values[0]])
    end_point = np.array([len(singular_values), singular_values[-1]])
    max_distance = 0
    elbow_point = 0
    for i, singular_value in enumerate(singular_values):
        current_point = np.array([i, singular_value])
        distance = norm(np.cross(start_point - end_point, start_point - current_point)) / \
            norm(end_point - start_point)
        if distance > max_distance:
            max_distance = distance
            elbow_point = i
    return elbow_point


def get_optimal_sequence_add(distance_matrix: np.ndarray, scenes_count: int) -> np.ndarray:
    """
    Divide shots into scenes regarding to H_add metrics.
    More info in paper: https://ieeexplore.ieee.org/abstract/document/7823628
    :return: indexes of the last shot of each scene
    """
    D = distance_matrix
    K = scenes_count
    N = len(D)
    C = np.zeros((N, K))
    J = np.zeros((N, K), dtype=int)

    for n in range(0, N):
        C[n, 0] = np.sum(D[n:, n:])

    for n in range(0, N):
        J[n, 0] = N - 1

    for k in range(1, K):
        for n in range(0, N):
            candidates = []
            for i in range(n, N):
                if i < N - 1:
                    C_prev = C[i + 1, k - 1]
                else:
                    C_prev = 0
                h_n_i = np.sum(D[n:i + 1, n:i + 1])
                candidate = h_n_i + C_prev
                candidates.append(candidate)
            candidates = np.array(candidates)
            C[n, k] = np.min(candidates)
            J[n, k] = np.where(candidates == C[n, k])[0][0] + n

    t = np.zeros((K,), dtype=int)
    t_prev = 0
    for i in range(0, K):
        if i == 0:
            t_prev = 0
        else:
            t_prev = t[i - 1]
        t[i] = J[t_prev, K - i - 1]
    return t


def get_intervals_from_borders(borders: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Convert scene borders to intervals
    
    :param borders: list of borders
    :return: list of interval tuples where first value - beginning of an interval, the second - end of an interval
    """
    intervals = []
    prev_border = 0
    for cur_border in borders:
        intervals.append((prev_border, cur_border))
        prev_border = cur_border
    return intervals

@benchmark
def get_scenes(img_path):
    """Returns start/end shots index pairs for each scene."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resnet = resnet101(weights='DEFAULT')
    for p in resnet.parameters():
        p.requires_grad = False

    modules = list(resnet.children())[:-2] # Delete last 2 layers as long as we don't have classification problem
    model = nn.Sequential(*modules).to(device)
    model.eval()

    shots_imgs_names = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    shots_imgs = []

    for img_name in shots_imgs_names:
        img = np.array(Image.open(f"{img_path+img_name}"), dtype=np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        shots_imgs.append(img)

    shots_vecs = [model(img.to(device)).view(-1).cpu().numpy() for img in shots_imgs]

    dist_mat = squareform(pdist(shots_vecs))

    scenes_count = estimate_scenes_count(dist_mat)

    last_shot_of_each_scene = get_optimal_sequence_add(dist_mat, scenes_count)
    scenes_shots_idx = get_intervals_from_borders(last_shot_of_each_scene)

    return scenes_shots_idx


@benchmark
def cut_video(video_path: str, scenes_save_path: str, scenes_intervals: list)-> None:

    for i, interval in enumerate(scenes_intervals):
        video = VideoFileClip(video_path).subclip(interval[0], interval[1])
        video.write_videofile(f"{scenes_save_path}Scene_{i+1}.mp4", fps=24)


@benchmark
def txt_cap2voice_cap(captions, VOICE_CAPTION_DIR):

    for i, caption in enumerate(captions):
        myobj = gTTS(text=caption, lang='en', slow=False) 
        myobj.save(VOICE_CAPTION_DIR+f"Scene_{i+1}_cap.mp3") 


def scene_composite(scenes_path: str, voices_path: str, resulting_path: str, scene_name: str) -> None:
    """
    Composite scene with corresponding voice caption.
    """

    scene = VideoFileClip(f'{scenes_path + scene_name}.mp4')
    audio_background = AudioFileClip(f'{voices_path + scene_name}_cap.mp3')
    
    scene = concatenate_videoclips([scene.subclip(0, audio_background.duration).fx(afx.volumex, 0.5), scene.subclip(audio_background.duration, scene.duration)])

    final_audio = CompositeAudioClip([audio_background, scene.audio]) 
    final_clip = scene.set_audio(final_audio)
    final_clip.write_videofile(f'{resulting_path}Resulting_{scene_name}.mp4')


def wrapped_scene_composition(*args, **kwargs):
    return benchmark(scene_composite)(*args, **kwargs)


@benchmark
def concatenate_scenes(scenes_path: str, video_path: str, scenes_count: int) -> None:
    res_scenes = [VideoFileClip(f'{scenes_path}Resulting_Scene_{i+1}.mp4') for i in range(scenes_count)]
    
    final_film = concatenate_videoclips(res_scenes)
    final_film.write_videofile("videos/Final.mp4")


def files_clear() -> None:
    """
    Dirs to clear: images, results, scenes, voice captions
    """

    folders = ['images', 'results', 'scenes', 'voice_captions']
    for folder in folders:
        for filename in listdir(folder):
            file_path = join(folder, filename)
            try:
                if isfile(file_path):
                    unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def export_worktime(worktime: List[int],  funcs: List[str]) -> None:

    colors = sns.color_palette('pastel')[0:5]

    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%\n{:d} sec".format(pct, absolute)

    plt.pie(worktime, labels = funcs, colors = colors, autopct=lambda pct: func(pct, worktime))

    today = datetime.datetime.today()

    plt.savefig(f'diagrams/{today.strftime("%Y-%m-%d-%H.%M.%S")}.jpg')
