import torch
import imageio.v3 as iio
import imageio
from PIL import Image
import os
import cv2
from transformers import CLIPTokenizerFast, T5TokenizerFast
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
import random
import numpy as np


def get_clip_token_indices(
    pipe,
    prompt,
    word,
    path
):
    if os.path.exists(path):
        tokenizer = CLIPTokenizerFast.from_pretrained(path)
    else:
        raise ValueError("Please enter the correct path to the tokenizer")

    prompt = [prompt]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = text_inputs["offset_mapping"].squeeze().cpu().numpy()

    token_indices = find_token_indices_for_word(word, prompt[0], offsets)
    return token_indices

def get_t5_token_indices(
    pipe,
    prompt,
    word,
    max_sequence_length: int = 256,
    is_flux=False,
):
    prompt = [prompt]
    tokenizer = pipe.tokenizer_2 if is_flux else pipe.tokenizer_3
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = text_inputs["offset_mapping"].squeeze().cpu().numpy()

    token_indices = find_token_indices_for_word(word, prompt[0], offsets)
    return token_indices

def get_cog_t5_token_indices(
    pipe,
    prompt,
    word,
    path,
    max_sequence_length: int = 256,
):
    prompt = [prompt]

    if os.path.exists(path):
        tokenizer = T5TokenizerFast.from_pretrained(path)
    else:
        raise ValueError(f"Please enter the correct path to the tokenizer: {path}")

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = text_inputs["offset_mapping"].squeeze().cpu().numpy()

    token_indices = find_token_indices_for_word(word, prompt[0], offsets)

    text_input_ids = text_inputs.input_ids

    prompt_tokens = []
    for i in range(len(prompt)):
        input_ids = text_input_ids[i].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        prompt_tokens.append({
            "original_prompt": prompt[i],
            "tokens": tokens,
            "token_ids": input_ids.tolist(),
        })
        print(f"Prompt: {prompt[i]}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {input_ids.tolist()}\n")
    return token_indices

def find_token_indices_for_word(word, prompt_text, offset_mapping):
    start_char = prompt_text.find(word)
    end_char = start_char + len(word)
    
    token_indices = []
    for idx, (offset_start, offset_end) in enumerate(offset_mapping):
        if not (offset_end <= start_char or offset_start >= end_char):
            token_indices.append(idx)
    
    return token_indices


@torch.no_grad()
def latent2image(pipe, latents, device, dtype, custom_shape=None):
    '''return: PIL.Image'''
    if hasattr(pipe, '_unpack_latents'):
        # default square
        if custom_shape is None:
          latents = pipe._unpack_latents(
              latents, 
              int((latents.shape[1] ** 0.5) * pipe.vae_scale_factor) * 2,
              int((latents.shape[1] ** 0.5) * pipe.vae_scale_factor) * 2,
              pipe.vae_scale_factor,
          )
        else:
          latents = pipe._unpack_latents(
              latents, 
              custom_shape[0], custom_shape[1],
              pipe.vae_scale_factor,
          )
        
    latents = latents.to(device).to(dtype)
    if pipe.vae.config.shift_factor is not None:
        # There was a bug here when submitting the code to the paper, as shown in https://github.com/DSL-Lab/UniEdit-Flow/issues/3
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    else:
        latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type='pil')[0]
    return image

def save_numpy_arrays_as_mp4(frames, fp, fps=8):
    """
    You need to install imageio to use this function

    `pip install imageio[ffmpeg,pyav]`
    """

    h, w, _ = frames[0].shape
    if h % 2 != 0 or w % 2 != 0:
        new_frames = []
        for frame in frames:
            new_frames.append(cv2.resize(frame, (w - w % 2, h - h % 2)))
        frames = new_frames

    with imageio.imopen(fp, "w" + "I", plugin="pyav", extension=".mp4", legacy_mode=True) as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": "17"}
        writer.write(frames)

def load_video_imageio(video_path):
    images = []
    for frame in iio.imiter(video_path, plugin="pyav"):
        images.append(Image.fromarray(frame))
    return images

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
