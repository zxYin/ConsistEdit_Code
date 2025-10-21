python run_synthesis_sd3.py \
    --src_prompt "a portrait of a woman in a red dress, realistic style, best quality" \
    --tgt_prompt "a portrait of a woman in a red dress, cartoon style, best quality" \
    --edit_object "dress" \
    --out_dir "output" \
    --alpha 0.3 \
    --no_mask \
    --model_path "/cpfs/_zixin/ckpts/stable-diffusion-3-medium-diffusers"