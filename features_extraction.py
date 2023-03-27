import cv2
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet101


def extract_features(video_path, total_frame=28):
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
            #transform images
            #print(RGBimage.shape)
            frames.append(data_transform(RGBimage))
            count += 1
        else:
            fail += 1
    vidObj.release()
    frames = torch.stack(frames)
    interval = count//total_frame
    frames = frames[range(0,interval*total_frame,interval)]
    
    with torch.no_grad():
        output_features = model(frames).unsqueeze(0)
    
    return output_features

