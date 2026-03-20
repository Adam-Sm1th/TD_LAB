import os
from kornia.augmentation.auto.rand_augment.ops import brightness
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
import math
import numpy as np
from pixelseal.videoseal.utils.display import save_img
from pixelseal.videoseal.evals.full import setup_model_from_checkpoint
from pixelseal.videoseal.evals.metrics import bit_accuracy, psnr, ssim
from pixelseal.videoseal.augmentation import Identity, JPEG


class PixelSealProvider():
    def __init__(self,device):
        self.model = setup_model_from_checkpoint('pixelseal')
        self.model.eval()
        self.model.compile()
        self.model.to(device)
        self.model.blender.scaling_w *= 1  # control watermark strength

    def encode(self, img):
        outputs = self.model.embed(img, is_video=False, lowres_attenuation=True)
        imgs_w = outputs["imgs_w"]
        msgs = outputs["msgs"]

        return imgs_w,msgs
    def decode(self,img, msgs):
        outputs_det = self.model.detect(img, is_video=False)
        acc = bit_accuracy(outputs_det["preds"][:, 1:], msgs).nanmean().item()
        print(acc)
        return outputs_det["preds"][:, 1:]



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open("C:\\Users\\Adam\\Desktop\\fft\\lena.png", "r").convert("RGB")
    n = PixelSealProvider(device)
    imgs_w, msgs = n.encode(img)
    n.decode(img,msgs)
