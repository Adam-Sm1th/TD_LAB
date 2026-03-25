from pixelseal_provider import PixelSealProvider #忽略爆红
from PIL import Image
import torch
import torchvision

if __name__ == "__main__":
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