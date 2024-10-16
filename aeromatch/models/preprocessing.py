import torch.nn as nn
from torchvision import transforms

# Format the image to [N, C, H, W] and pass through
def preprocess_img(img, img_size):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        #transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_torch = preprocess(img)
    return img_torch
