import os
from datetime import datetime as dt
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
import torch

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from cuda import cudart


# os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

class pth_onnx():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
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
                timesteps = torch.zeros(1, dtype=torch.int32).to("cuda")
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
                control_in = []
                control_names = []
                for i in range(13):
                    control_names.append("control_" + str(i))
                b=1
                h=32
                w=48
                for i in range(3):
                    temp = torch.zeros(b, 320, h, w, dtype=torch.float32).to("cuda")
                    control_in.append(temp)
                
                temp = torch.zeros(b, 320, h//2, w//2, dtype=torch.float32).to("cuda")
                control_in.append(temp)

                for i in range(2):
                    temp = torch.zeros(b, 640, h//2, w//2, dtype=torch.float32).to("cuda")
                    control_in.append(temp)

                temp = torch.zeros(b, 640, h//4, w//4, dtype=torch.float32).to("cuda")
                control_in.append(temp)

                for i in range(2):
                    temp = torch.zeros(b, 1280, h//4, w//4, dtype=torch.float32).to("cuda")
                    control_in.append(temp)

                for i in range(4):
                    temp = torch.zeros(b, 1280, h//8, w//8, dtype=torch.float32).to("cuda")
                    control_in.append(temp)
                
                dynamic_table = {'x' : {0 : 'B', 2 : 'H', 3 : 'W'},
                        'timesteps' : {0 : 'B'},
                        'context' : {0 : 'B'}}
                for i in range(13):
                    dynamic_table[control_names[i]] = {0:'bs'}#,2:'dim2',3:'dim3'}

                temp_model = getattr(self.model.model, v)
                onnxfile = "./unet.onnx"
                with torch.inference_mode(), torch.autocast("cuda"):
                    temp_model = temp_model.cuda()
                    x = torch.randn(1, 4, 32, 48, dtype=torch.float32, device='cuda')
                    timesteps = torch.tensor([951] , dtype=torch.int32, device='cuda')
                    context = torch.randn(1, 77, 768, dtype=torch.float32, device='cuda')
                    torch.onnx.export(temp_model, 
                    (x, timesteps, context, control_in), 
                    onnxfile, 
                    export_params=True, 
                    do_constant_folding=True, 
                    keep_initializers_as_inputs=True, 
                    opset_version=17, 
                    input_names=["x", "timesteps", "context"] + control_names, 
                    output_names=["output"], 
                    dynamic_axes=dynamic_table
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
