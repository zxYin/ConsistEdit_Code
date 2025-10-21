import os
import torch
import argparse

from diffusers import FluxPipeline

from consistEdit.global_var import GlobalVars
from consistEdit.attention_control import (
    regiter_attention_editor_diffusers_flux,
    regiter_attention_editor_diffusers_flux_mask,
)
from consistEdit.solver import register_flux_solver
from consistEdit.utils import get_t5_token_indices, setup_seed


def consistent_synthesis(model_path, alpha, out_dir, src_prompt, tgt_prompt, edit_object, use_mask=True, use_old_mask=False):
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.enable_sequential_cpu_offload()
    register_flux_solver(pipe)

    seed = 42
    setup_seed(seed)

    prompts = [
        src_prompt,
        tgt_prompt,
    ]
    token_words = edit_object

    width = 1024
    height = 1024
    steps = 28
    vae_scale_factor = 8
    channel_num = 16
    bs = len(prompts)
    start_code = torch.randn([1, channel_num, width // vae_scale_factor, height // vae_scale_factor], device=device, dtype=dtype)
    start_code = start_code.expand(bs, -1, -1, -1)
    start_code = pipe._pack_latents(start_code, bs, channel_num, width // vae_scale_factor, height // vae_scale_factor)

    GlobalVars.WIDTH = width // 8 // 2
    GlobalVars.HEIGHT = height // 8 // 2
    GlobalVars.TEXT_LENGTH = 512

    if use_mask:
        regiter_attention_editor_diffusers_flux_mask(pipe, use_old_version=use_old_mask)
    else:
        regiter_attention_editor_diffusers_flux(pipe)

    os.makedirs(out_dir, exist_ok=True)
    GlobalVars.TEST_QK_STEP = int(steps * alpha)

    if use_mask:
        GlobalVars.TOKEN_IDS = [x for x in get_t5_token_indices(pipe, prompts[0], token_words, is_flux=True)]
        GlobalVars.GENERATE_MASK = True
        GlobalVars.MASK_OUTPUT_PATH = os.path.join(out_dir, "mask.png")
        image_ori = pipe.denoise(prompts[:1], width=width, height=height, latents=start_code[:1], guidance_scale=7.5, num_inference_steps=steps).images

    GlobalVars.GENERATE_MASK = False
    image_ori = pipe.denoise(prompts, width=width, height=height, latents=start_code, guidance_scale=7.5, num_inference_steps=steps).images

    image_ori[0].save(os.path.join(out_dir, f"source.png"))
    image_ori[1].save(os.path.join(out_dir, f"target.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_prompt', type=str, default="a portrait of a woman in a red dress in a forest, best quality")
    parser.add_argument('--tgt_prompt', type=str, default="a portrait of a woman in a yellow dress in a forest, best quality")
    parser.add_argument('--edit_object', type=str, default="dress")
    parser.add_argument('--out_dir', type=str, default="output")
    parser.add_argument('--no_mask', action="store_true")
    parser.add_argument('--use_old_mask', action="store_true")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model_path', type=str, default="/cpfs/_zixin/ckpts/FLUX.1-dev")
    args = parser.parse_args()

    dtype = torch.float16
    torch.cuda.set_device(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    use_mask = not args.no_mask
    use_old_mask = args.use_old_mask

    with torch.amp.autocast(dtype=dtype, device_type="cuda"):
        consistent_synthesis(args.model_path, args.alpha, args.out_dir, args.src_prompt, args.tgt_prompt, args.edit_object, use_mask, use_old_mask)
