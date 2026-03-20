import os

from kornia.augmentation.auto.rand_augment.ops import brightness
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
import math
import numpy as np
from imageattacker import ImageAttacker
from videoseal.utils.display import save_img
from videoseal.evals.full import setup_model_from_checkpoint
from videoseal.evals.metrics import bit_accuracy, psnr, ssim
from videoseal.augmentation import Identity, JPEG
import wandb

# wandb.init(
#     project="pixelseal-test",  # 项目名称，网页上按此分类
#     name="robust-experiment", # 本次实验的具体名字
#     config={                   # 记录你的超参数，方便以后对比
#         "scaling_w": 1.0,
#         "model_type": "pixelseal-base",
#         "num_imgs": 1
#     }
# )

attacks = {
    # JPEG quality: 100 (无损) → 5 (极度损耗), 步长 5
    "jpeg": list(range(100, 0, -5)),

    # contrast: 1.0 (原图) → 2.0 (增强) 以及 1.0 → 0.1 (减弱)
    # 建议测试从强对比到弱对比
    "contrast": [i / 10 for i in range(20, 0, -1)],

    # brightness: 同上，从 2.0 倍亮度到 0.1 倍亮度
    "brightness": [i / 10 for i in range(20, 0, -1)],

    # scaling: 1.0 (原尺寸) → 0.1 (缩小到10%), 步长 0.1
    "scaling": [i / 10 for i in range(10, 0, -1)],

    # --- 新增部分 ---

    # blur: 半径从 0 (原图) 到 10 (极度模糊), 步长 0.5
    "blur": [i / 2 for i in range(0, 21)],

    # rotation: 角度从 0 到 180 度, 步长 10 (或者测较小的范围如 0-45)
    # 水印通常对旋转非常敏感，建议先测 [0, 5, 10, 15, 20, 25, 30, 45]
    "rotation": list(range(150, -150, -5)),
}

to_tensor = torchvision.transforms.ToTensor()
to_pil = torchvision.transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Directory containing images
num_imgs = 10
assets_dir = "assets/imgs"
base_output_dir = "outputs"
output_dir = os.path.join(base_output_dir, 'pixelseal')
os.makedirs(base_output_dir, exist_ok=True)

#model setting
model = setup_model_from_checkpoint('pixelseal')
model.eval()
model.compile()
model.to(device)
model.blender.scaling_w *= 1 #control watermark strength

#load testImages
files = [f for f in os.listdir(assets_dir) if f.endswith(".png") or f.endswith(".jpg")]
files = [os.path.join(assets_dir, f) for f in files]
files = files[:num_imgs]

# --- 3. 初始化大图布局 (2行3列) ---
num_attacks = len(attacks)
cols = 3
rows = math.ceil(num_attacks / cols)
fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
axes = axes.flatten()  # 展开成一维，方便循环调用

# --- 4. 主循环计算 ---
for i, (attackName, attackList) in enumerate(attacks.items()):
    accList = []
    ax = axes[i]  # 获取当前的子图坐标系

    for item in tqdm(attackList, desc=f"{attackName} attack"):
        accListTemp = []
        for file in files:
            # 数据处理与 Embed
            imgs = Image.open(file, "r").convert("RGB")
            imgs = to_tensor(imgs).unsqueeze(0).float()

            with torch.no_grad():
                outputs = model.embed(imgs, is_video=False, lowres_attenuation=True)
                imgs_w = outputs["imgs_w"]
                msgs = outputs["msgs"]

                # 动态调用攻击
                attacker = ImageAttacker(imgs_w)
                attack_func = getattr(attacker, attackName)
                imgs_aug = attack_func(item)

                # 检测与准确率计算
                outputs_det = model.detect(imgs_aug, is_video=False)
                acc = bit_accuracy(outputs_det["preds"][:, 1:], msgs).nanmean().item()

            accListTemp.append(acc)
            del outputs, outputs_det, imgs, imgs_w

        accList.append(np.mean(accListTemp))

    # --- 5. 在子图中绘图 ---
    x = np.array(attackList)
    y = np.array(accList)

    ax.plot(x, y, marker='o', linestyle='-', linewidth=2, label="Bit Acc")

    # 标注最大值
    idx_max = np.argmax(y)
    ax.scatter(x[idx_max], y[idx_max], color='red', s=60, zorder=5)
    ax.text(x[idx_max], y[idx_max], f' Max:{y[idx_max]:.3f}', color='red', fontweight='bold')

    # 标注最小值
    idx_min = np.argmin(y)
    ax.scatter(x[idx_min], y[idx_min], color='green', s=60, zorder=5)
    ax.text(x[idx_min], y[idx_min], f' Min:{y[idx_min]:.3f}', color='green', fontweight='bold', verticalalignment='top')

    # 子图装饰
    ax.set_title(f"Attack: {attackName}", fontsize=15)
    ax.set_xlabel("Strength / Parameter")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

# 如果攻击不满 6 个，隐藏剩余的空白子图
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# --- 6. 整体展示 ---
plt.tight_layout()  # 自动调整布局防止重叠
plt.suptitle("Robustness Analysis Under Different Image Attacks", fontsize=20, y=1.02)
plt.savefig(os.path.join(output_dir, "robustness_report.png"), bbox_inches='tight', dpi=300)
plt.show()  # 所有跑完后一次性显示

#Test


del model
torch.cuda.empty_cache()

