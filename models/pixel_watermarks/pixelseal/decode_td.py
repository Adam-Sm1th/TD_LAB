from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from videoseal.evals.full import setup_model_from_checkpoint
from videoseal.evals.metrics import bit_accuracy, psnr, ssim

if __name__ == "__main__":
    to_tensor = torchvision.transforms.ToTensor()
    to_pil = torchvision.transforms.ToPILImage()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model_from_checkpoint("pixelseal")
    model.eval()
    model.compile()
    model.to(device)
    indices = [1, 2, 5, 7, 9]
    input_msg = torch.zeros(256).unsqueeze(0)
    input_msg[0, indices] = 1

    img = Image.open("1.png")
    img = to_tensor(img).unsqueeze(0).float()

    outputs = model.detect(img, is_video=False)
    metrics = {
        "bit_accuracy": bit_accuracy(
            outputs["preds"][:, 1:],
            input_msg
        ).nanmean().item(),
    }
    print(metrics)