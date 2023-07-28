echo "preprocess"
python3 2onnx.py

trtexec \
--onnx=./controlnet.onnx \
--optShapes=x:1x4x32x48,hint:1x3x256x384,timesteps:1,context:1x77x768 \
--workspace=10240 \
--saveEngine=./dynamic/controlnet.plan \
--fp16 --skipInference \
--verbose \
--device=7

trtexec \
--onnx=./unet.onnx \
--optShapes=x:1x4x32x48,timesteps:1,context:1x77x768 \
--workspace=10240 \
--saveEngine=./dynamic/unet.plan \
--fp16 --skipInference \
--verbose \
--device=7

trtexec \
--onnx=./clip.onnx \
--optShapes=input_ids:1x77 \
--workspace=10240 \
--saveEngine=./dynamic/clip.plan \
--fp16 --skipInference \
--verbose \
--device=7

trtexec \
--onnx=./vae.onnx \
--optShapes=latent:1x4x32x48 \
--workspace=10240 \
--saveEngine=./dynamic/vae.plan \
--fp16 --skipInference \
--verbose \
--device=7
