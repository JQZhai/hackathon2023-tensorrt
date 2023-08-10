echo "preprocess"
python3 2onnx.py

trtexec \
--onnx=./controlnet.onnx \
--workspace=10240 \
--saveEngine=./controlnet.plan \
--fp16 \
--builderOptimizationLevel=5 \
--optShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \
--minShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \
--maxShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768

trtexec \
--onnx=./clip.onnx \
--optShapes=input_ids:1x77 \
--minShapes=input_ids:1x77 \
--maxShapes=input_ids:1x77 \
--workspace=10240 \
--saveEngine=./clip.plan \
--builderOptimizationLevel=5

trtexec \
--onnx=./vae.onnx \
--workspace=10240 \
--saveEngine=./vae.plan \
--builderOptimizationLevel=5 \
--optShapes=latent:1x4x32x48 \
--minShapes=latent:1x4x32x48 \
--maxShapes=latent:1x4x32x48


trtexec \
--onnx=./unet.onnx --saveEngine=unet.plan  \
--builderOptimizationLevel=5 --fp16 \
--optShapes=x:1x4x32x48,timesteps:1,context:1x77x768,control_0:1x320x32x48,control_1:1x320x32x48,control_2:1x320x32x48,control_3:1x320x16x24,control_4:1x640x16x24,control_5:1x640x16x24,control_6:1x640x8x12,control_7:1x1280x8x12,control_8:1x1280x8x12,control_9:1x1280x4x6,control_10:1x1280x4x6,control_11:1x1280x4x6,control_12:1x1280x4x6 \
--minShapes=x:1x4x32x48,timesteps:1,context:1x77x768,control_0:1x320x32x48,control_1:1x320x32x48,control_2:1x320x32x48,control_3:1x320x16x24,control_4:1x640x16x24,control_5:1x640x16x24,control_6:1x640x8x12,control_7:1x1280x8x12,control_8:1x1280x8x12,control_9:1x1280x4x6,control_10:1x1280x4x6,control_11:1x1280x4x6,control_12:1x1280x4x6 \
--maxShapes=x:1x4x32x48,timesteps:1,context:1x77x768,control_0:1x320x32x48,control_1:1x320x32x48,control_2:1x320x32x48,control_3:1x320x16x24,control_4:1x640x16x24,control_5:1x640x16x24,control_6:1x640x8x12,control_7:1x1280x8x12,control_8:1x1280x8x12,control_9:1x1280x4x6,control_10:1x1280x4x6,control_11:1x1280x4x6,control_12:1x1280x4x6
