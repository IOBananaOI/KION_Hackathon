from typing import Tuple, List
from os import listdir
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

import cv2
import random
import unicodedata


class Utils:
    '''
    Generic utility functions that our model and dataloader would require

    '''
   
    @staticmethod
    def set_seed(seed):
        '''
          For reproducibility 
        '''
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def unicodeToAscii(s):
        '''
        Turn a Unicode string to plain ASCII, 
        Thanks to https://stackoverflow.com/a/518232/2809427
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def target_tensor_to_caption(voc,target):
        '''
        Convert target tensor to Caption
        '''
        gnd_trh = []
        lend = target.size()[1]
        for i in range(lend):
            tmp = ' '.join(voc.index2word[x.item()] for x in target[:,i])
            gnd_trh.append(tmp)
        return gnd_trh

    @staticmethod
    def maskNLLLoss(inp, target, mask, device):
        '''
        Masked cross-entropy loss calculation; 
        refers: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
        '''
        inp = inp.squeeze(0)
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp.squeeze(0), 1, target.view(-1, 1)).squeeze(1).float())
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()
 
    @staticmethod
    def FrameCapture(video_path, video_name):
        '''
        Function to extract frames
        For MSVD Sample every 10th frame
        '''
        
        #video_path = video_path_dict[video_name]
        # Path to video file 
        video_path = video_path+video_name  #Changes
        vidObj = cv2.VideoCapture(video_path) 
        count = 0
        fail = 0
        # checks whether frames were extracted 
        success = 1
        frames = []
        while success: 
            # OpenCV Uses BGR Colormap
            success, image = vidObj.read() 
            try:
                RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if count%10 == 0:            #Sample 1 frame per 10 frames
                    frames.append(RGBimage)
                count += 1
            except:
                fail += 1
        vidObj.release()
        if count > 80:
            frames = frames[:81]
        return np.stack(frames[:-1]),count-1, fail


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


def cut_video(video_path: str, scenes_save_path: str, scenes_intervals: list)-> None:

    for i, interval in enumerate(scenes_intervals):
        video = VideoFileClip(video_path).subclip(interval[0], interval[1])
        video.write_videofile(f"{scenes_save_path}Scene{i+1}.mp4", fps=24)


def scenes_composite(scenes_path: str, voices_path: str, resulting_path: str, scenes_count: int) -> None:
    """
    Composite scenes with corresponding voice captions.
    """

    for i in range(scenes_count):
        my_clip = VideoFileClip(f'{scenes_path}Scene{i+1}.mp4')

        audio_background = AudioFileClip(f'{voices_path}Scene_cap_{i+1}.mp3')
        my_clip = concatenate_videoclips([my_clip.subclip(0, audio_background.duration).fx(afx.volumex, 0.5), my_clip.subclip(audio_background.duration, my_clip.duration)])

        final_audio = CompositeAudioClip([audio_background, my_clip.audio]) 
        final_clip = my_clip.set_audio(final_audio)
        final_clip.write_videofile(f'{resulting_path}resulting_scene_{i+1}.mp4')


def concatenate_scenes(scenes_path: str, video_path: str, scenes_count: int) -> None:
    res_scenes = [VideoFileClip(f'{scenes_path}resulting_scene_{i+1}.mp4') for i in range(scenes_count)]
    
    final_film = concatenate_videoclips(res_scenes)
    final_film.write_videofile("videos/Final.mp4")
