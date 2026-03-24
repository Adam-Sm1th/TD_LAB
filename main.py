import os
import torch
import pandas as pd
import argparse
from utils.imprint_utils import validate
from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.utils import get_detection_threshold, check_if_detection_successful
from utils.pipe import pipe_utils
from utils.prompt_utils import PROMPTS_SD_LIST, PROMPTS_I2P_LIST
from utils.utils import set_random_seed
from torch import Tensor
import torchvision
from PIL import Image
from pixelseal_provider import PixelSealProvider  # 忽略爆红
from utils.image_utils import torch_to_PIL, PIL_to_torch


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
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 20 for FLUX
    parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 3.5 for FLUX

    # attacker model
    parser.add_argument("--modelid_attacker", type=str,
                        default="Manojb/stable-diffusion-2-1-base")  # changed from stabilityai/stable-diffusion-2-1-base, because stabilityAI took down SD2.1
    parser.add_argument("--scheduler_attacker", type=str, default="DDIM")
    parser.add_argument("--num_inference_steps_attacker", type=int, default=50)
    parser.add_argument("--guidance_scale_attacker", type=float, default=7.5)

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


def reprompting_attack(img: Tensor, args):
    attacker_prompt = PROMPTS_I2P_LIST[
        args.attacker_prompt_index] if args.attacker_prompt is None else args.attacker_prompt
    pipe_provider_attacker = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_attacker,
                                                          resolution=args.resolution,
                                                          device=DEVICE,
                                                          eager_loading=False,
                                                          disable_tqdm=True
                                                          )  # base model

    with torch.no_grad():
        res_2 = pipe_provider_attacker.invert_images(images=img,
                                                     num_inference_steps=args.num_inference_steps_attacker)

        if args.resample:
            recovered_zT = wm_provider.wiggle_latents(res_2["zT_torch"].clone())
            recovered_zT = recovered_zT.to(dtype=pipe_provider_attacker.get_dtype())
        else:
            recovered_zT = res_2["zT_torch"].clone()

        res_3 = pipe_provider_attacker.generate(prompts=attacker_prompt,
                                                num_inference_steps=args.num_inference_steps_attacker,
                                                guidance_scale=args.guidance_scale_attacker,
                                                latents=recovered_zT,
                                                )
        reprompting_tensor = res_3["images_torch"]

    pipe_provider_attacker.stash_pipe()
    return reprompting_tensor


if __name__ == "__main__":
    # device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = fetch_ages()
    set_random_seed(args.seed)

    # --------------------------------------------------------------- generate a image with semantic watemark ----------------------------------------------------------------------
    # provider model
    print("generate a image with semantic watemark")
    pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                        resolution=args.resolution,
                                                        schedulers_name=args.scheduler_target,
                                                        unet_id_or_checkpoint_dir=None,
                                                        lora_checkpoint_dir=None,
                                                        device=DEVICE,
                                                        eager_loading=True if "FLUX" in args.modelid_target else False,
                                                        disable_tqdm=True
                                                        )  # finetuned model

    # wm provider
    wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), **vars(args))
    wm_initial_results = wm_provider.get_wm_latents()
    wm_zT = wm_initial_results["zT_torch"]
    # generate a benign image
    target_prompt = PROMPTS_SD_LIST[args.target_prompt_index] if args.target_prompt is None else args.target_prompt
    res_1 = pipe_provider_target.generate(prompts=target_prompt,
                                          num_inference_steps=args.num_inference_steps_target,
                                          guidance_scale=args.guidance_scale_target,
                                          latents=wm_zT)
    gs_pil = res_1["images_PIL"][0]
    gs_tensor = res_1["images_torch"]
    gs_pil.save("gs_pil.png")

    with torch.no_grad():
        # retrieve zT
        zT_retrieved = pipe_provider_target.invert_images(gs_pil, num_inference_steps=args.num_inference_steps_target)[
            "zT_torch"]

    # watermark test
    accuracy_results = wm_provider.get_accuracies(zT_retrieved)
    bit_accuracy = accuracy_results["bit_accuracies"][0] if "bit_accuracies" in accuracy_results else 0.0
    print(f"GSwatermark bit accuracy = {bit_accuracy}")
    pipe_provider_target.stash_pipe()

    # --------------------------------------------------------------- add pixel watermark ----------------------------------------------------------------------
    batch_size = 1
    message_length = 256
    random_message = torch.randint(0, 2, (batch_size, message_length)).to(DEVICE)

    # read image
    to_tensor = torchvision.transforms.ToTensor()

    # init pixel provider
    n = PixelSealProvider(DEVICE)

    # emmbed watermark
    imgs_w = n.encode(gs_tensor, random_message)

    # detect watermark
    decoded_msg, acc = n.decode(imgs_w, random_message)
    print(f"pixelwatermark acc = {acc}")

    pixel_gs_pil = torch_to_PIL(imgs_w)[0]
    pixel_gs_pil.save("pixel_gs_pil.png")

    # --------------------------------------------------------------- reprompt attack ----------------------------------------------------------------------
    print(f"start reprompt attack")
    reprompting_tensor = reprompting_attack(imgs_w, args)
    reprompting_pil = torch_to_PIL(reprompting_tensor)[0]
    reprompting_pil.save("reprompt.png")
    # --------------------------------------------------------------- reprompt acc ----------------------------------------------------------------------
    pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                        resolution=args.resolution,
                                                        schedulers_name=args.scheduler_target,
                                                        unet_id_or_checkpoint_dir=None,
                                                        lora_checkpoint_dir=None,
                                                        device=DEVICE,
                                                        eager_loading=True if "FLUX" in args.modelid_target else False,
                                                        disable_tqdm=True
                                                        )  # finetuned model
    # detect pixel watermark
    decoded_msg, acc = n.decode(reprompting_tensor, random_message)
    print(f"reprompt pixel watermark bit accuracy = {acc}")

    with torch.no_grad():
        # retrieve zT
        zT_retrieved = \
        pipe_provider_target.invert_images(reprompting_tensor, num_inference_steps=args.num_inference_steps_target)[
            "zT_torch"]

    pipe_provider_target.stash_pipe()
    # detect senmantic watermark
    accuracy_results = wm_provider.get_accuracies(zT_retrieved)
    bit_accuracy = accuracy_results["bit_accuracies"][0] if "bit_accuracies" in accuracy_results else 0.0
    print(f"reprompt GSwatermark bit accuracy = {bit_accuracy}")






