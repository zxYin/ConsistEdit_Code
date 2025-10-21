import os 
import gc
import cv2
import json
import torch
import argparse
import numpy as np
from diffusers import StableDiffusion3Pipeline

from consistEdit.utils import latent2image
from consistEdit.global_var import GlobalVars
from consistEdit.solver import register_sd3_solver
from consistEdit.attention_control import regiter_attention_editor_diffusers_sd3_mask
from consistEdit.utils import get_t5_token_indices, get_clip_token_indices, setup_seed


if __name__ == "__main__":
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
        parser.add_argument('--data_path', type=str, default="/data/open_source/") # the editing category that needed to run
        parser.add_argument('--output_path', type=str, default="result_output") # the editing category that needed to run
        parser.add_argument('--model_path', type=str, default="/cpfs/_zixin/ckpts/stable-diffusion-3-medium-diffusers") # the editing category that needed to run
        parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
        args = parser.parse_args()
        
        rerun_exist_images=args.rerun_exist_images
        data_path=args.data_path
        output_path=args.output_path
        edit_category_list=args.edit_category_list
        
        with open(f"{data_path}/mapping_file.json", "r") as f:
            editing_instruction = json.load(f)

        model_path = args.model_path
        tokenizer_path = os.path.join(model_path, "tokenizer")
        dtype = torch.float16
        device = "cuda"
        pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)

        register_sd3_solver(pipe)
        regiter_attention_editor_diffusers_sd3_mask(pipe)

        items = editing_instruction.items()

        for key, item in items:
            
            if item["editing_type_id"] not in edit_category_list: continue
            
            blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []

            if item["editing_type_id"] == "3":
                token_words = item["original_prompt"][item["original_prompt"].find("[")+1:item["original_prompt"].find("]")]
            else:
                token_words = blended_word[0] if len(blended_word) > 0 else None

            GlobalVars.NO_V = (item["editing_type_id"] == "9")

            if item["editing_type_id"] in ["6", "7"]:
                GlobalVars.TEST_QK_STEP = 28
            else:
                GlobalVars.TEST_QK_STEP = 8

            original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

            editing_instruction = item["editing_instruction"]

            output_name = "ours"
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path, output_name)).replace(".jpg", ".png")

            mask_path = present_image_save_path.replace("annotation_images", "mask_images")
            source_path = present_image_save_path.replace("annotation_images", "source_images")

            if not os.path.exists(os.path.dirname(present_image_save_path)): os.makedirs(os.path.dirname(present_image_save_path))
            if not os.path.exists(os.path.dirname(mask_path)): os.makedirs(os.path.dirname(mask_path))
            if not os.path.exists(os.path.dirname(source_path)): os.makedirs(os.path.dirname(source_path))

            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                setup_seed(42)

                GlobalVars.MAP_SAVER = {}

                torch.cuda.empty_cache()
                gc.collect()

                prompts = [original_prompt, editing_prompt]

                vae_scale_factor = 8
                channel_num = 16
                width = 1024
                height = 1024

                start_code = torch.randn([1, channel_num, width // vae_scale_factor, height // vae_scale_factor], device=device, dtype=torch.float16)
                start_code = start_code.repeat(len(prompts), 1, 1, 1)

                GlobalVars.MASK_OUTPUT_PATH = mask_path

                # if not os.path.exists(mask_path):
                if True:
                    if GlobalVars.NO_V:
                        all_white_mask = np.zeros((64, 64)) * 255
                        cv2.imwrite(mask_path, all_white_mask)
                    else:
                        GlobalVars.TOKEN_IDS = get_clip_token_indices(pipe, original_prompt, token_words, tokenizer_path) + ([x + 77 for x in get_t5_token_indices(pipe, original_prompt, token_words)])
                        GlobalVars.GENERATE_MASK = True
                        image_latents = pipe.denoise(prompts[:1], latents=start_code[:1], guidance_scale=7.5, output_type="latent").images

                GlobalVars.GENERATE_MASK = False
                image_latents = pipe.denoise(prompts, latents=start_code, guidance_scale=7.5, output_type="latent").images

                source_image = latent2image(pipe, image_latents[:1], device, dtype)
                source_image.save(source_path)

                edit_image = latent2image(pipe, image_latents[1:], device, dtype)

                edit_image.save(present_image_save_path)
                print(f"finish")
            else:
                print(f"skip image [{image_path}]")
        
                