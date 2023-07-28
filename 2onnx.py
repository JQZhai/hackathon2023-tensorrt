import os
from datetime import datetime as dt
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
import torch

import sys
sys.path.append("../")
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

torch.cuda.set_device(7)
from cuda import cudart


os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

class pth_onnx():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('/home/player/ControlNet/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()

        self.state_dict = {
                    "clip": "cond_stage_model",
                    "control_net": "control_model",
                    "unet": "diffusion_model",
                    "vae": "first_stage_model"
                }

        for k, v in self.state_dict.items():
            if k == "control_net":
                temp_model = getattr(self.model, v)
                onnxfile = "./controlnet.onnx"
                H = 256
                W = 384
                context = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
                x = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
                hint = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
                timesteps = torch.zeros(1, dtype=torch.int64).to("cuda")
                dynamic_table = {'x' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                                'hint' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                                'timesteps' : {0 : 'bs'},
                                'context' : {0 : 'bs'}}
                output_names = []
                for i in range(13):
                    output_names.append("out_"+ str(i))
                for i in range(13):
                    dynamic_table[output_names[i]] = {0 : "bs"}

                torch.onnx.export(temp_model, 
                (x, hint, timesteps, context), 
                onnxfile, 
                export_params=True, 
                do_constant_folding=True, 
                keep_initializers_as_inputs=True, 
                opset_version=17, 
                input_names=["x", "hint", "timesteps", "context"], 
                output_names=output_names,
                dynamic_axes=dynamic_table
                )
                
            if k == "unet":
                temp_model = getattr(self.model.model, v)
                onnxfile = "./unet.onnx"
                x = torch.randn(1, 4, 32, 48, device='cuda')
                timesteps = torch.zeros(1, device='cuda') + 500
                context = torch.randn(1, 77, 768, device='cuda')
                # control = 13个张量的列表，shape[1,320,32,48]
                torch.onnx.export(temp_model, 
                (x, timesteps, context), 
                onnxfile, 
                export_params=True, 
                do_constant_folding=True, 
                keep_initializers_as_inputs=True, 
                opset_version=17, 
                input_names=["x", "timesteps", "context"], 
                output_names=["output"], 
                dynamic_axes={'x' : {0 : '2B', 2 : 'H', 3 : 'W'},
                            'context' : {0 : '2B'},
                            'output' : {0 : '2B', 2: 'H', 3: 'W'}}
                )
            
            if k == "clip":
                temp_model = getattr(self.model, v)
                model = temp_model.transformer
                self.tokenizer = temp_model.tokenizer
                onnxfile = "./clip.onnx"
                input_ids = torch.zeros(1, 77, dtype= torch.int32, device='cuda')
                torch.onnx.export(model, 
                input_ids, 
                onnxfile, 
                export_params=True, 
                do_constant_folding=True, 
                keep_initializers_as_inputs=True, 
                opset_version=17, 
                input_names=["input_ids"], 
                output_names=["text_embeddings", 'pooler_output'], 
                dynamic_axes={'input_ids':{0:'B'},
                                'text_embeddings':{0:'B'}}
                )

            if k == "vae":
                temp_model = getattr(self.model, v)
                model = temp_model
                model.forward = model.decode 
                onnxfile = "./vae.onnx"
                latent = torch.randn(1, 4, 32, 48, dtype= torch.float32, device='cuda')
                torch.onnx.export(model, 
                latent, 
                onnxfile, 
                export_params=True, 
                do_constant_folding=True, 
                keep_initializers_as_inputs=True, 
                opset_version=17, 
                input_names=["latent"], 
                output_names=["images"], 
                dynamic_axes={'latent': {0: 'B', 2: 'H', 3: 'W'},
                                'images': {0: 'B', 2: '8H', 3: '8W'}}
                )
ins = pth_onnx()
ins.initialize()
