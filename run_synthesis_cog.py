import os
import torch
import argparse
import numpy as np
from PIL import Image
import imageio.v3 as iio

from diffusers import CogVideoXPipeline

from consistEdit.global_var import GlobalVars
from consistEdit.attention_control import (
    regiter_attention_editor_diffusers_cog,
    regiter_attention_editor_diffusers_cog_mask,
)
from consistEdit.solver import register_cog_solver
from consistEdit.utils import get_cog_t5_token_indices, save_numpy_arrays_as_mp4, setup_seed


def load_video_imageio(video_path):
    images = []
    for frame in iio.imiter(video_path, plugin="pyav"):
        images.append(Image.fromarray(frame))
    return images


def consistent_synthesis(model_path, alpha, out_dir, src_prompt, tgt_prompt, edit_object, use_mask=True, use_old_mask=False):
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.enable_sequential_cpu_offload()
    register_cog_solver(pipe)

    seed = 42
    setup_seed(seed)

    prompts = [
        src_prompt,
        tgt_prompt,
    ]
    token_words = edit_object

    width = 720
    height = 480
    steps = 50
    vae_scale_factor = 8
    channel_num = 16
    num_frames = 49
    T = (num_frames - 1) // 4 + 1
    bs = len(prompts)

    GlobalVars.WIDTH = width // 8 // 2
    GlobalVars.HEIGHT = height // 8 // 2
    GlobalVars.TEXT_LENGTH = 226

    os.makedirs(out_dir, exist_ok=True)

    start_code = torch.randn([1, T, channel_num, height // vae_scale_factor, width // vae_scale_factor], device=device, dtype=dtype)
    start_code = start_code.expand(bs, -1, -1, -1, -1)

    if use_mask:
        regiter_attention_editor_diffusers_cog_mask(pipe, use_old_version=use_old_mask)
    else:
        regiter_attention_editor_diffusers_cog(pipe)

    GlobalVars.TEST_QK_STEP = int(steps * alpha)

    if use_mask:
        tokenizer_path = os.path.join(model_path, "tokenizer")
        GlobalVars.TOKEN_IDS = get_cog_t5_token_indices(pipe, prompts[0], token_words, tokenizer_path)
        GlobalVars.GENERATE_MASK = True
        GlobalVars.MASK_OUTPUT_PATH = os.path.join(out_dir, "mask")
        if not os.path.exists(GlobalVars.MASK_OUTPUT_PATH):
            os.makedirs(GlobalVars.MASK_OUTPUT_PATH, exist_ok=True)
        videos = pipe.denoise(prompts[:1], latents=start_code[:1], guidance_scale=6, num_inference_steps=steps).frames

    GlobalVars.GENERATE_MASK = False
    videos = pipe.denoise(prompts, latents=start_code, guidance_scale=6, num_inference_steps=steps).frames

    save_numpy_arrays_as_mp4(np.clip(videos[0], 0, 255), os.path.join(out_dir, f"source.mp4"))
    save_numpy_arrays_as_mp4(np.clip(videos[1], 0, 255), os.path.join(out_dir, f"target.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_prompt', type=str, default="a portrait of a woman in a red dress in a forest, best quality")
    parser.add_argument('--tgt_prompt', type=str, default="a portrait of a woman in a yellow dress in a forest, best quality")
    parser.add_argument('--edit_object', type=str, default="dress")
    parser.add_argument('--out_dir', type=str, default="output")
    parser.add_argument('--no_mask', action="store_true")
    parser.add_argument('--use_old_mask', action="store_true")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model_path', type=str, default="/cpfs/_zixin/ckpts/CogVideoX-2b")
    args = parser.parse_args()

    dtype = torch.float16
    torch.cuda.set_device(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    use_mask = not args.no_mask
    use_old_mask = args.use_old_mask

    with torch.no_grad():
        consistent_synthesis(args.model_path, args.alpha, args.out_dir, args.src_prompt, args.tgt_prompt, args.edit_object, use_mask, use_old_mask)
