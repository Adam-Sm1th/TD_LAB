
import torch

from .schedulers.scheduling_ddim import DDIMScheduler
from .schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

from diffusers import DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

from .SD_provider import SDPipeProvider
from .SDXL_provider import SDXLPipeProvider
from .PixArt_provider import PixArtPipeProvider



# Map model id onto pipe. Add more if needed
PIPE_PROVIDERS = {
    'Manojb/stable-diffusion-2-1-base': SDPipeProvider,  # changed from stabilityai/stable-diffusion-2-1-base, because stabilityAI took down SD2.1
    'stabilityai/stable-diffusion-xl-base-1.0': SDXLPipeProvider,
    'PixArt-alpha/PixArt-Sigma-XL-2-512-MS': PixArtPipeProvider,
    # FLUX see below
    }


SCHEDULER_CLASSES = {
     "DDIM": (DDIMScheduler, DDIMInverseScheduler),
     "DPM": (DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler),
     "Euler": (None, None),  # special case for Flux
}


def get_pipe_provider(pretrained_model_name_or_path: str,
                      resolution: int,
                      unet_id_or_checkpoint_dir: str = None,
                      lora_checkpoint_dir: str = None,
                      vae_id: str = None,
                      zero_unet: bool = False,
                      device: torch.device = torch.device("cuda"),
                      eager_loading: bool = False,
                      schedulers_name: str = "DDIM",
                      **kwargs):

    # 1. 确保半精度（注入到 kwargs）
    if 'torch_dtype' not in kwargs:
        kwargs['torch_dtype'] = torch.float16

    # 2. 特殊模型动态加载
    if "FLUX" in pretrained_model_name_or_path:
         from .Flux_provider import FluxPipeProvider
         PIPE_PROVIDERS['black-forest-labs/FLUX.1-dev'] = FluxPipeProvider

    # 3. 初始化 provider 实例
    pipe_provider_class = PIPE_PROVIDERS[pretrained_model_name_or_path]
    provider_instance = pipe_provider_class(
                             pretrained_model_name_or_path=pretrained_model_name_or_path,
                             resolution=resolution,
                             unet_id_or_checkpoint_dir=unet_id_or_checkpoint_dir,
                             lora_checkpoint_dir=lora_checkpoint_dir,
                             vae_id=vae_id,
                             zero_unet=zero_unet,
                             device=device,
                             eager_loading=eager_loading,
                             scheduler_classes=SCHEDULER_CLASSES[schedulers_name],
                             **kwargs)

    # 4. 核心优化：在实例创建后，且 pipe 加载后执行（顺序关键！）
    # 检查实例是否有 pipe 属性（通常在 provider 内部 self.pipe 存储 diffusers 管道）
    if hasattr(provider_instance, 'pipe') and provider_instance.pipe is not None:
        if "PixArt" in pretrained_model_name_or_path or "XL" in pretrained_model_name_or_path:
            # 开启 CPU 卸载：模型各组件在计算时才进入 GPU
            provider_instance.pipe.enable_model_cpu_offload()
            # 开启 VAE 分块与切片：解决你之前的 vae_encode 爆显存问题
            provider_instance.pipe.vae.enable_tiling()
            provider_instance.pipe.vae.enable_slicing()
    
    return provider_instance