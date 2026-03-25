import os
import gc
import torch
import pandas as pd
import argparse
import torchvision
from PIL import Image
from torch import Tensor

# 设置显存分配策略，防止碎片化 (解决 64MB 申请失败的问题)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from utils.imprint_utils import validate
from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.utils import get_detection_threshold, check_if_detection_successful
from utils.pipe import pipe_utils
from utils.prompt_utils import PROMPTS_SD_LIST, PROMPTS_I2P_LIST
from utils.utils import set_random_seed
from pixelseal_provider import PixelSealProvider  # 忽略爆红
from utils.image_utils import torch_to_PIL, PIL_to_torch

def flush():
    """强制清理 GPU 显存缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def fetch_ages():
    # args
    parser = argparse.ArgumentParser(description="reprompt", parents=[gs_parser, tr_parser])
    parser.add_argument("--out_dir", type=str, default="out/reprompt/")
    
    # prompts
    parser.add_argument("--target_prompt_index", type=int, default=0, choices=list(range(len(PROMPTS_SD_LIST))))
    parser.add_argument("--target_prompt", type=str, default=None)
    parser.add_argument("--attacker_prompt_index", type=int, default=0, choices=list(range(len(PROMPTS_I2P_LIST))))
    parser.add_argument("--attacker_prompt", type=str, default=None)

    # target model
    parser.add_argument("--modelid_target",
                        type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        choices=["stabilityai/stable-diffusion-xl-base-1.0", "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
                                 "black-forest-labs/FLUX.1-dev"])
    parser.add_argument("--scheduler_target", type=str, default="DDIM")
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)
    parser.add_argument("--num_inference_steps_target", type=int, default=50)

    # attacker model
    parser.add_argument("--modelid_attacker", type=str,
                        default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--scheduler_attacker", type=str, default="DDIM")
    parser.add_argument("--num_inference_steps_attacker", type=int, default=50)
    parser.add_argument("--guidance_scale_attacker", type=float, default= 7.5)

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--wm_type",
                        type=str,
                        default="GS",
                        choices=[wm.name for wm in WmProviders])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--test_image_path", type=str, default="test.jpg")
    
    args = parser.parse_args()
    return args

def reprompting_attack(img: Tensor, args, device):
    # attacker_prompt = PROMPTS_I2P_LIST[
        # args.attacker_prompt_index] if args.attacker_prompt is None else args.attacker_prompt

    attacker_prompt = "a girl eat a "
    pipe_provider_attacker = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_attacker,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        disable_tqdm=True
    )

    with torch.no_grad():
        res_2 = pipe_provider_attacker.invert_images(images=img,
                                                      num_inference_steps=args.num_inference_steps_attacker)

        if args.resample:
            # 注意：wm_provider 需要在外部定义或作为参数传入，这里假设已通过 args 逻辑处理
            # 为了代码健壮性，这里仅做示意逻辑
            recovered_zT = res_2["zT_torch"].clone() 
        else:
            recovered_zT = res_2["zT_torch"].clone()

        res_3 = pipe_provider_attacker.generate(prompts=attacker_prompt,
                                                num_inference_steps=args.num_inference_steps_attacker,
                                                guidance_scale=args.guidance_scale_attacker,
                                                latents=recovered_zT)
        reprompting_tensor = res_3["images_torch"].detach().clone()

    # 彻底清理攻击模型
    pipe_provider_attacker.stash_pipe()
    del pipe_provider_attacker
    flush()
    
    return reprompting_tensor


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = fetch_ages()
    # set_random_seed(args.seed)

    # ------------------ 阶段 1: 生成带水印图像 (Target Model) ------------------
    print(">>> 正在生成带语义水印的图像...")
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        schedulers_name=args.scheduler_target,
        device=DEVICE,
        eager_loading=True if "FLUX" in args.modelid_target else False,
        disable_tqdm=True
    )

    wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), **vars(args))
    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]

    target_prompt = PROMPTS_SD_LIST[args.target_prompt_index] if args.target_prompt is None else args.target_prompt
    res_1 = pipe_provider_target.generate(prompts=target_prompt,
                                          num_inference_steps=args.num_inference_steps_target,
                                          guidance_scale=args.guidance_scale_target,
                                          latents=wm_zT)
    
    gs_pil = res_1["images_PIL"][0]
    gs_tensor = res_1["images_torch"].detach().clone()
    gs_pil.save("gs_pil.png")

    with torch.no_grad():
        inv_res = pipe_provider_target.invert_images(gs_pil, num_inference_steps=args.num_inference_steps_target)
        zT_retrieved = inv_res["zT_torch"].detach().clone()
        del inv_res

    accuracy_results = wm_provider.get_accuracies(zT_retrieved)
    print(f"原始语义水印准确率: {accuracy_results.get('bit_accuracies', [0.0])[0]}")

    # 彻底释放 Target 模型以腾出空间给攻击模型
    pipe_provider_target.stash_pipe()
    del pipe_provider_target
    flush()

    # ------------------ 阶段 2: 叠加像素水印 (PixelSeal) ------------------
    print(">>> 正在叠加像素水印...")
    batch_size = 1
    message_length = 256
    random_message = torch.randint(0, 2, (batch_size, message_length)).to(DEVICE)
    
    n = PixelSealProvider(DEVICE)
    imgs_w = n.encode(gs_tensor, random_message)
    
    # 验证像素水印
    _, acc = n.decode(imgs_w, random_message)
    print(f"叠加后像素水印准确率: {acc}")
    torch_to_PIL(imgs_w)[0].save("pixel_gs_pil.png")

    # ------------------ 阶段 3: Reprompt 攻击 (Attacker Model) ------------------
    print(">>> 开始 Reprompt 攻击...")
    reprompting_tensor = reprompting_attack(imgs_w, args, DEVICE)
    torch_to_PIL(reprompting_tensor)[0].save("reprompt.png")

    # ------------------ 阶段 4: 最终检测 (重新加载 Target Model) ------------------
    print(">>> 重新加载目标模型进行检测...")
    pipe_provider_target = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        schedulers_name=args.scheduler_target,
        device=DEVICE,
        disable_tqdm=True
    )

    # 检测像素水印
    _, pixel_acc = n.decode(reprompting_tensor, random_message)
    print(f"攻击后像素水印准确率: {pixel_acc}")

    # 检测语义水印 (需要再次 Inversion)
    with torch.no_grad():
        inv_res_final = pipe_provider_target.invert_images(reprompting_tensor, num_inference_steps=args.num_inference_steps_target)
        zT_final = inv_res_final["zT_torch"].detach().clone()
        del inv_res_final

    final_accuracy_results = wm_provider.get_accuracies(zT_final)
    final_bit_acc = final_accuracy_results.get("bit_accuracies", [0.0])[0]
    print(f"攻击后语义水印准确率: {final_bit_acc}")

    # 任务结束，收尾清理
    pipe_provider_target.stash_pipe()
    del pipe_provider_target
    flush()
    print(">>> 实验完成。")