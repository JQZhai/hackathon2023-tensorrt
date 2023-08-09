from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
import tensorrt as trt

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        self.model.engine = {}
        with open("./controlnet.plan", 'rb') as f:
            engine_str = f.read()
        self.model.engine['control_net'] = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        
        with open("./unet.plan", 'rb') as f:
            engine_str = f.read()
        self.model.engine['unet'] = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        
        with open("./clip.plan", 'rb') as f:
            engine_str = f.read()
        self.model.engine['clip'] = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        
        with open("./vae.plan", 'rb') as f:
            engine_str = f.read()
        self.model.engine['vae'] = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)

        self.model.context = {}
        for i in self.model.engine:
            self.model.context[i] = self.model.engine[i].create_execution_context()
            if i == 'control_net':
                self.model.context[i].set_binding_shape(0, (1, 4, 32, 48))
                self.model.context[i].set_binding_shape(1, (1, 3, 256, 384))
                self.model.context[i].set_binding_shape(2, (1, ))
                self.model.context[i].set_binding_shape(3, (1, 77, 768))
            if i == 'unet':
                self.model.context[i].set_binding_shape(0, (1, 4, 32, 48))
                self.model.context[i].set_binding_shape(1, (1, ))
                self.model.context[i].set_binding_shape(2, (1, 77, 768))
                self.model.context[i].set_binding_shape(3, (1, 320, 32, 48))
                self.model.context[i].set_binding_shape(4, (1, 320, 32, 48))
                self.model.context[i].set_binding_shape(5, (1, 320, 32, 48))
                self.model.context[i].set_binding_shape(6, (1, 320, 16, 24))
                self.model.context[i].set_binding_shape(7, (1, 640, 16, 24))
                self.model.context[i].set_binding_shape(8, (1, 640, 16, 24))
                self.model.context[i].set_binding_shape(9, (1, 640, 8, 12))
                self.model.context[i].set_binding_shape(10,(1, 1280, 8, 12))
                self.model.context[i].set_binding_shape(11,(1, 1280, 8, 12))
                self.model.context[i].set_binding_shape(12, (1, 1280, 4, 6))
                self.model.context[i].set_binding_shape(13, (1, 1280, 4, 6))
                self.model.context[i].set_binding_shape(14, (1, 1280, 4, 6))  
                self.model.context[i].set_binding_shape(14, (1, 1280, 4, 6))              
            if i == 'clip':
                self.model.context[i].set_binding_shape(0, (1, 77))
            if i == 'vae':
                self.model.context[i].set_binding_shape(0, (1, 4, 32, 48))
        print("load engine finished")


    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            ddim_steps=10
            samples, intermediates = self.ddim_sampler.sample(ddim_steps=10, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results
