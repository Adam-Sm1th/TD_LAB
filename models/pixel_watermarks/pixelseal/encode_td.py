from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from videoseal.evals.full import setup_model_from_checkpoint

if __name__ == "__main__":
    to_tensor = torchvision.transforms.ToTensor()
    to_pil = torchvision.transforms.ToPILImage()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model_from_checkpoint("pixelseal")
    model.eval()
    model.compile()
    model.to(device)

    img = Image.open("0.bmp")
    img = to_tensor(img).unsqueeze(0).float()
    img_clone = img.clone()

    h, w = img.shape[-2], img.shape[-1]
    indices = [1, 2, 5, 7, 9]
    input_msg = torch.zeros(256).unsqueeze(0)
    input_msg[0, indices] = 1

    for i in tqdm(range(h // 256), desc="Processing Rows"):
        for j in tqdm(range(w // 256), desc="Processing Cols", leave=False):
            y_s, y_e = i * 256, min((i + 1) * 256, h)
            x_s, x_e = j * 256, min((j + 1) * 256, w)
            chunk = img[:,:,i * 256: (i + 1) * 256, j * 256: (j + 1) * 256]
            outputs = model.embed(chunk, msgs=input_msg, is_video=False, lowres_attenuation=True)
            img[:, :, y_s:y_e, x_s:x_e] = outputs["imgs_w"]

    diff = (img - img_clone)
    diff_save = to_pil(diff.squeeze())
    diff_save.save("diff.png")
    img_save = to_pil(img.squeeze())
    img_save.save("0_save.bmp")

