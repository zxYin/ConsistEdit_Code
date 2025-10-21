import os
import torch
import argparse
from PIL import Image

from diffusers import StableDiffusion3Pipeline

from consistEdit.global_var import GlobalVars
from consistEdit.attention_control import (
    regiter_attention_editor_diffusers_sd3_mask_real,
)
from consistEdit.solver import register_sd3_solver
from consistEdit.scheduler import UniInvEulerScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from consistEdit.utils import get_t5_token_indices, get_clip_token_indices, latent2image, setup_seed


def consistent_synthesis(model_path, alpha, out_dir, src_prompt, tgt_prompt, edit_object, source_image_path, use_old_mask=False):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
    invert_scheduler = UniInvEulerScheduler.from_pretrained(model_path, subfolder='scheduler')
    register_sd3_solver(pipe)

    seed = 42
    setup_seed(seed)

    source_prompts = [src_prompt]
    target_prompts = [src_prompt, tgt_prompt]
    token_words = edit_object
    width = 1024
    height = 1024
    steps = 28
    bs = len(source_prompts)
    GlobalVars.WIDTH = width // 8 // 2
    GlobalVars.HEIGHT = height // 8 // 2
    GlobalVars.TEXT_LENGTH = 333

    source_image = Image.open(source_image_path).resize((width, height)).convert("RGB")
    source_tensor = pipe.image_processor.preprocess(source_image).to(dtype=dtype, device=device)
    image_latent = pipe.vae.encode(source_tensor)['latent_dist'].mean
    image_latent = (image_latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    regiter_attention_editor_diffusers_sd3_mask_real(pipe, use_old_version=use_old_mask)

    os.makedirs(out_dir, exist_ok=True)
    GlobalVars.TEST_QK_STEP = int(steps * alpha)

    tokenizer_path = os.path.join(model_path, "tokenizer")
    clip_indices = get_clip_token_indices(pipe, source_prompts[0], token_words, tokenizer_path)
    t5_indices = [x + 77 for x in get_t5_token_indices(pipe, source_prompts[0], token_words)]
    GlobalVars.TOKEN_IDS = clip_indices + t5_indices
    GlobalVars.MASK_OUTPUT_PATH = os.path.join(out_dir, "mask.png")
    pipe.scheduler = invert_scheduler

    start_latent = pipe.invert(source_prompts, latents=image_latent, guidance_scale=1, num_inference_steps=steps, output_type="latent").images
    torch.save(start_latent, os.path.join(out_dir, "latent.pt"))

    bs = len(target_prompts)
    start_latent = start_latent.expand(bs, -1, -1, -1)
    pipe.scheduler = scheduler

    image_latents = pipe.denoise(target_prompts, latents=start_latent, num_inference_steps=steps, guidance_scale=1, output_type="latent").images

    source_image = latent2image(pipe, image_latents[:1], device, dtype)
    source_image.save(os.path.join(out_dir, f"source.png"))
    edit_image = latent2image(pipe, image_latents[1:2], device, dtype)
    edit_image.save(os.path.join(out_dir, f"target.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_prompt', type=str, default="a girl with a red hat and red t-shirt is sitting in a park, best quality")
    parser.add_argument('--tgt_prompt', type=str, default="a girl with a yellow hat and red t-shirt is sitting in a park, best quality")
    parser.add_argument('--edit_object', type=str, default="hat")
    parser.add_argument('--source_image_path', type=str, default="assets/red_hat_girl.png")
    parser.add_argument('--out_dir', type=str, default="output")
    parser.add_argument('--use_old_mask', action="store_true")
    # Due to the information leakage in inversion, we set a small alpha for real image editing.
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--model_path', type=str, default="/cpfs/_zixin/ckpts/stable-diffusion-3-medium-diffusers")
    args = parser.parse_args()

    dtype = torch.float16
    torch.cuda.set_device(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    use_old_mask = args.use_old_mask

    with torch.amp.autocast(dtype=dtype, device_type="cuda"):
        with torch.no_grad():
            consistent_synthesis(args.model_path, args.alpha, args.out_dir, args.src_prompt, args.tgt_prompt, args.edit_object, args.source_image_path, use_old_mask)
