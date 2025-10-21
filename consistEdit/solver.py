
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps, FluxPipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3PipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput

from consistEdit.global_var import GlobalVars

def register_sd3_solver(model):
    def denoise_raw(self, invert):
        @torch.no_grad()
        def call(
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            prompt_3: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 7.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
            max_sequence_length: int = 256,
            mu: Optional[float] = None,
        ):
            GlobalVars.reset_global_vars()
            GlobalVars.IS_INVERSE = invert
            GlobalVars.TOTAL_STEPS = num_inference_steps
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._joint_attention_kwargs = joint_attention_kwargs
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            lora_scale = (
                self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            # 4. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 5. Prepare timesteps
            scheduler_kwargs = {}
            if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
                _, _, height, width = latents.shape
                image_seq_len = (height // self.transformer.config.patch_size) * (
                    width // self.transformer.config.patch_size
                )
                mu = calculate_shift(
                    image_seq_len,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 4096),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.16),
                )
                scheduler_kwargs["mu"] = mu
            elif mu is not None:
                scheduler_kwargs["mu"] = mu
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            # 7. Denoising loop
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    if t == timesteps[-1]:
                        GlobalVars.SAVE_MASK_TRIGGER = True
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        # timestep=timestep/1000,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    GlobalVars.CUR_STEP += 1
            
            if output_type == "latent":
                image = latents

            else:
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusion3PipelineOutput(images=image)

        return call

    call_func = denoise_raw
    model.denoise = call_func(model, False)
    model.invert = call_func(model, True)

def register_flux_solver(model):
    def denoise_raw(self, invert):
        @torch.no_grad()
        def call(
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt: Union[str, List[str]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            true_cfg_scale: float = 1.0,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 3.5,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_sequence_length: int = 512,
        ):
            GlobalVars.reset_global_vars()
            GlobalVars.IS_INVERSE = invert
            GlobalVars.TOTAL_STEPS = num_inference_steps

            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            lora_scale = (
                self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            )
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
            )
            do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            if do_true_cfg:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

            # 4. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels // 4
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 5. Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            # if invert:
            #     timesteps = torch.flip(timesteps, dims=(0,))
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            # 6. Denoising loop
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    if t == timesteps[-1]:
                        GlobalVars.SAVE_MASK_TRIGGER = True
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    if do_true_cfg:
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    GlobalVars.CUR_STEP += 1

            self._current_timestep = None

            if output_type == "latent":
                image = latents
            else:
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)


            # Offload all models
            self.maybe_free_model_hooks()

            return FluxPipelineOutput(images=image)

        return call

    call_func = denoise_raw
    model.denoise = call_func(model, False)
    model.invert = call_func(model, True)

def register_cog_solver(model):
    def denoise_ddim(self, invert):
        @torch.no_grad()
        def call(
            prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: Optional[List[int]] = None,
            guidance_scale: float = 6,
            num_videos_per_prompt: int = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: str = "pil",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            max_sequence_length: int = 226,
        ): 
            GlobalVars.reset_global_vars()
            GlobalVars.IS_INVERSE = invert
            GlobalVars.TOTAL_STEPS = num_inference_steps

            height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
            width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
            num_frames = num_frames or self.transformer.config.sample_frames

            num_videos_per_prompt = 1

            self._guidance_scale = guidance_scale
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            # 2. Default call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            # timesteps += 1
            if invert:
                timesteps = torch.flip(timesteps, dims=(0,))
            self._num_timesteps = len(timesteps)

            # 5. Prepare latents
            latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

            # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
            patch_size_t = self.transformer.config.patch_size_t
            additional_frames = 0
            if patch_size_t is not None and latent_frames % patch_size_t != 0:
                additional_frames = patch_size_t - latent_frames % patch_size_t
                num_frames += additional_frames * self.vae_scale_factor_temporal

            latent_channels = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Create rotary embeds if required
            image_rotary_emb = (
                self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
                if self.transformer.config.use_rotary_positional_embeddings
                else None
            )

            # 8. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                # for DPM-solver++
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    if i == len(timesteps) - 1:
                        GlobalVars.SAVE_MASK_TRIGGER = True
                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float()

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * noise_pred

                    a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
                    b_t = alpha_prod_t_prev**0.5 - alpha_prod_t**0.5 * a_t
                    latents = a_t * latents + b_t * pred_original_sample

                    latents = latents.to(prompt_embeds.dtype)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                    GlobalVars.CUR_STEP += 1

            self._current_timestep = None

            if not output_type == "latent":
                # Discard any padding frames that were added for CogVideoX 1.5
                latents = latents[:, additional_frames:]
                video = self.decode_latents(latents)
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            else:
                video = latents

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (video,)

            return CogVideoXPipelineOutput(frames=video)
        return call
    
    call_func = denoise_ddim
    model.denoise = call_func(model, False)
    model.invert = call_func(model, True)