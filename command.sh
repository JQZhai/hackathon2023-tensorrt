# ../../TensorRT-8.6.1.6/bin/trtexec \
# --onnx=./controlnet.onnx \
# --workspace=10240 \
# --saveEngine=controlnet.plan \
# --fp16 --skipInference \
# --verbose \
# --device=7

# ../../TensorRT-8.6.1.6/bin/trtexec \
# --onnx=./unet/unet.onnx \
# --workspace=10240 \
# --saveEngine=unet.plan \
# --fp16 --skipInference \
# --verbose \
# --device=7

# ../../TensorRT-8.6.1.6/bin/trtexec \
# --onnx=./clip.onnx \
# --workspace=10240 \
# --saveEngine=clip.plan \
# --fp16 --skipInference \
# --verbose \
# --device=7

# ../../TensorRT-8.6.1.6/bin/trtexec \
# --onnx=./vae.onnx \
# --workspace=10240 \
# --saveEngine=vae.plan \
# --fp16 --skipInference \
# --verbose \
# --device=7

# --minShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \
# --optShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \
# --maxShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \

../../TensorRT-8.6.1.6/bin/trtexec \
--onnx=./dynamic/controlnet.onnx \
--optShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \
--workspace=10240 \
--saveEngine=./dynamic/controlnet.plan \
--fp16 --skipInference \
--verbose \
--device=7

../../TensorRT-8.6.1.6/bin/trtexec \
--onnx=./unet/unet.onnx \
--optShapes=x:1x4x32x48,timesteps:1,context:1x77x768 \
--workspace=10240 \
--saveEngine=./dynamic/unet.plan \
--fp16 --skipInference \
--verbose \
--device=7

../../TensorRT-8.6.1.6/bin/trtexec \
--onnx=./dynamic/clip.onnx \
--optShapes=input_ids:1x77 \
--workspace=10240 \
--saveEngine=./dynamic/clip.plan \
--fp16 --skipInference \
--verbose \
--device=7

../../TensorRT-8.6.1.6/bin/trtexec \
--onnx=./dynamic/vae.onnx \
--optShapes=latent:1x4x32x48 \
--workspace=10240 \
--saveEngine=./dynamic/vae.plan \
--fp16 --skipInference \
--verbose \
--device=7
