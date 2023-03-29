import os
# from features_extraction import extract_features

import warnings
warnings.filterwarnings('ignore')

from config import Path
from dictionary import Vocabulary

from config import ConfigSALSTM
from models.SA_LSTM.model import SALSTM

import cv2
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet101


def extract_features(video_path, frames_count_lim=150):
    device = torch.device('cpu')

    model = nn.Sequential(*list(resnet101(weights='DEFAULT').children())[:-1])
    model = model.to(device)
    model.eval()

    data_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((299,299)),transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225],inplace=True)])

    vidObj = cv2.VideoCapture(video_path) 
    count = 0
    success = 1
    frames = []
    fail = 0
    while success: 
        # OpenCV Uses BGR Colormap
        success, image = vidObj.read() 
        if success:
            RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if count%10 == 0:            #Sample 1 frame per 10 frames
                    frames.append(data_transform(RGBimage))
            count += 1
        else:
            fail += 1
    vidObj.release()

    if count > frames_count_lim:
        frames = frames[:frames_count_lim]

    frames = torch.stack(frames)
    
    with torch.no_grad():
        output_features = model(frames).unsqueeze(0)
    
    return output_features, len(frames)


def get_captions(scenes_path: str, scenes_count) -> list:

    #create SALSTM config object
    cfg = ConfigSALSTM(opt_encoder=True)
    cfg.dataset = 'msrvtt'
    cfg.dropout = 0.5
    cfg.opt_param_init = False

    #creation of path object
    path = Path(cfg,os.getcwd())
    #Vocabulary object, 
    voc = Vocabulary(cfg)
    voc.load()

    min_count = 5 #remove all words below count min_count
    voc.trim(min_count=min_count)

    #Model object

    model = SALSTM(voc,cfg,path)
    model.load_state_dict(torch.load("epochs_81.pth"))
    model.eval()

    features_list = []
    for i in range(scenes_count):
        features, features_count = extract_features(f"Scene{i+1}.mp4")
        features = features.view(1, features_count, 2048)
        features_list.append(features)

    features_batch = torch.cat(features_list, dim=0)

    beam_txt_captions = model.BeamDecoding(features_batch.to(cfg.device), 10)

    return beam_txt_captions
