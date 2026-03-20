import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import io

class ImageAttacker:
    def __init__(self, tensor_img: torch.Tensor):
        """
        :param tensor_img: Tensor 形状为 [1, 3, H, W], 值域 [0, 1]
        """
        self.device = tensor_img.device
        # 转回 PIL 进行攻击
        self.pil_img = transforms.ToPILImage()(tensor_img.squeeze(0).cpu())
        self.w, self.h = self.pil_img.size

    def _to_tensor(self, pil_img):
        return transforms.ToTensor()(pil_img).unsqueeze(0).to(self.device)

    def jpeg(self, quality=50):
        buf = io.BytesIO()
        self.pil_img.save(buf, format="JPEG", quality=quality)
        return self._to_tensor(Image.open(buf))

    def contrast(self, factor=0.8):
        return self._to_tensor(ImageEnhance.Contrast(self.pil_img).enhance(factor))

    def brightness(self, factor=0.8):
        return self._to_tensor(ImageEnhance.Brightness(self.pil_img).enhance(factor))

    def scaling(self, scale=0.5):
        small = self.pil_img.resize((int(self.w*scale), int(self.h*scale)), Image.BILINEAR)
        return self._to_tensor(small.resize((self.w, self.h), Image.BILINEAR))

    def blur(self, radius=2.0):
        return self._to_tensor(self.pil_img.filter(ImageFilter.GaussianBlur(radius)))

    def rotation(self, degree=15):
        """
        :param degree: 旋转角度，例如 15 代表顺时针旋转 15 度
        """
        # resample=Image.BILINEAR 保证旋转后的平滑度
        # expand=False 保证输出尺寸依然是 (self.w, self.h)，超出部分会被截断或留黑边
        rotated = self.pil_img.rotate(degree, resample=Image.BILINEAR, expand=False)
        return self._to_tensor(rotated)

    def gaussian_noise(self, std=0.05):
        tensor = self._to_tensor(self.pil_img)
        noise = torch.randn_like(tensor) * std
        noisy_img = torch.clamp(tensor + noise, 0, 1)
        return noisy_img