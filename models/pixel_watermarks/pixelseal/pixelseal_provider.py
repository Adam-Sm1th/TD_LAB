from PIL import Image
import torch
import torchvision
from videoseal.evals.full import setup_model_from_checkpoint
from videoseal.evals.metrics import bit_accuracy, psnr, ssim
from videoseal.augmentation import Identity, JPEG


class PixelSealProvider():
    def __init__(self,device):
        self.model = setup_model_from_checkpoint('pixelseal')
        self.model.eval()
        self.model.compile()
        self.model.to(device)
        self.model.blender.scaling_w *= 1  # control watermark strength

    def convert_preds_to_binary(self, preds: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        按照 bit_accuracy 的逻辑，将原始预测张量转换为 [1, 256] 的 0/1 张量

        Args:
            preds: 原始预测张量，形状为 [1, 257, H, W]
            threshold: 判定阈值，默认 0.0
        Returns:
            torch.Tensor: 形状为 [1, 256] 的 0/1 张量
        """
        # 1. 判定正负 (b, 257, h, w)
        # 逻辑：正数变为 True (1)，负数/零变为 False (0)
        binary_preds = (preds > threshold).float()

        # 2. 空间维度压缩 (投票机制)
        # 逻辑：计算所有像素点 (H, W) 的平均值
        # 如果大部分像素是 1，平均值就会 > 0.5
        if binary_preds.dim() == 4:
            # 对最后两个维度（高和宽）求平均值，结果维度变为 [1, 257]
            voted_preds = binary_preds.mean(dim=(-2, -1))
        else:
            voted_preds = binary_preds

        # 3. 二次二值化 (少数服从多数)
        # 逻辑：均值 > 0.5 的位定为 1，否则为 0
        final_bits = (voted_preds > 0.5).int()

        # 4. 对齐切片 (解决 257 -> 256 的问题)
        # 在语义水印中，第 0 位通常是 Detection Bit，后 256 位是消息
        if final_bits.shape[1] == 257:
            final_bits = final_bits[:, 1:]  # 舍弃第 0 位，保留后 256 位

        return final_bits

    def encode(self, img: torch.Tensor, msg: torch.Tensor):
        outputs = self.model.embed(imgs=img, msgs=msg ,is_video=False, lowres_attenuation=True)
        imgs_w = outputs["imgs_w"]
        return imgs_w

    def decode(self, img: torch.Tensor, msg: torch.Tensor=None):
        outputs_det = self.model.detect(img, is_video=False)
        if(msg is not None):
            acc = bit_accuracy(outputs_det["preds"][:, 1:], msg).nanmean().item()
        else:
            acc = -1
        decoded_msg = self.convert_preds_to_binary(outputs_det["preds"][:, 1:],0)
        return decoded_msg, acc



if __name__ == '__main__':
    #初始化嵌入信息
    batch_size = 1
    message_length = 256
    random_message = torch.randint(0, 2, (batch_size, message_length))
    print(random_message)

    #读取图片
    to_tensor = torchvision.transforms.ToTensor()
    img = Image.open("C:\\Users\\Adam\\Desktop\\fft\\lena.png", "r").convert("RGB")
    img = to_tensor(img).unsqueeze(0).float()

    #创建编解码器类
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = PixelSealProvider(device)

    #嵌入水印
    imgs_w = n.encode(img, random_message)

    #提取水印
    decoded_msg, acc = n.decode(imgs_w, random_message)
    print(acc)
    print(decoded_msg)

