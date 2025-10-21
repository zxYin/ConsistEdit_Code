# coding=utf-8
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import torch
import numpy as np

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DDIMInverseScheduler
from diffusers.utils import BaseOutput


@dataclass
class UniInvEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


@dataclass
class UniInvDDIMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class UniInvEulerScheduler(FlowMatchEulerDiscreteScheduler):
    zero_initial=False
    alpha=1

    def set_hyperparameters(self, zero_initial=False, alpha=1):
        self.zero_initial = zero_initial
        self.alpha = alpha
    
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps
        else:
            self.num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        
        # timesteps
        timesteps = sigmas * self.config.num_train_timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1).to(sigmas)])
        self.timesteps = timesteps.flip(dims=[0]).to(device=device)

        # sigmas
        sigmas = torch.cat([sigmas, torch.zeros(1).to(sigmas)])
        self.sigmas = sigmas.flip(dims=[0]).to(device=device)
        
        # empty dt and derivative
        self.sample = None
        
        # zero_initial
        if self.zero_initial:
            self.timesteps = self.timesteps[1: ]
            self.sigmas = self.sigmas[1: ]
            self.sample = 'placeholder'
            self.first_sigma = 0
            
        # alpha, early stop
        if self.alpha < 1:
            inv_steps = math.floor(self.alpha * self.num_inference_steps)
            skip_steps = self.num_inference_steps - inv_steps
            self.timesteps = self.timesteps[: -skip_steps]
            self.sigmas = self.sigmas[: -skip_steps]

        self._step_index = 0
        self._begin_index = 0
        
        
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[UniInvEulerSchedulerOutput, Tuple]:
        
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
            
        sample = sample.to(torch.float32)

        if self.sample is None:
            # just for the first step
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            
            derivative = model_output               # v_0 = f(t=0, x_0)
            dt = sigma_next - sigma                 # sigma_{t + \Delta t} - sigma_t

            # store for correction
            self.sample = sample                    # Z_0
            
            prev_sample = sample + derivative * dt
            prev_sample = prev_sample.to(model_output.dtype)
        else:
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]
            
            if isinstance(self.sample, str):
                # for zero_initial
                sigma = self.first_sigma
                self.sample = sample
                
            derivative = model_output
            dt = sigma_next - sigma

            sample = self.sample

            self.sample = sample + dt * derivative

            if (self.step_index + 1) < len(self.sigmas):
                sigma_next_next = self.sigmas[self.step_index + 1]
                dt_next = sigma_next_next - sigma_next
                
                prev_sample = self.sample + dt_next * derivative
            else:
                # end loop
                prev_sample = self.sample
            prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return UniInvEulerSchedulerOutput(prev_sample=prev_sample)
