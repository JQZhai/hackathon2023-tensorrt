polygraphy run ./my_file/unet.onnx --onnxrt --trt --workspace 28G \
--save-engine=unet.onnx_fp16.plan --atol 1e-3 --rtol 1e-3 --verbose \
--input-shape 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--trt-min-shapes 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--trt-opt-shapes 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--trt-max-shapes 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--onnx-outputs mark all  --trt-outputs mark all \
--fp16 --save-inputs ./input.json --save-outputs ./output.json

polygraphy run ./unet.onnx \
--onnxrt -v --workspace=28G --fp16 \
--input-shape 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--onnx-outputs mark all \
--save-inputs onnx_input.json --save-outputs onnx_res.json

polygraphy debug precision ../LeViT-128S.onnx \
-v --fp16 --workspace 28G --no-remove-intermediate --log-file ./log_file.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
-p fp32 --mode bisect --dir forward --show-output \
--artifacts ./polygraphy_debug.engine --art-dir ./art-dir \
--check \
polygraphy run polygraphy_debug.engine \
--trt --load-outputs onnx_res.json --load-inputs onnx_input.json \
--abs 1e-2 -v --rel 1e-2


polygraphy convert -v --model-type onnx --input-shapes 'input_0:[1,3,224,224]' \
--shape-inference --seed 7 --load-inputs ../fp16/onnx_input.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
--int8 --workspace 30G --calibration-cache ./cal_trt.bin -o LeViT-128S.plan \
../LeViT-128S.onnx

polygraphy run unet.plan \
--trt -v --workspace=28G  --model-type engine  \
--input-shape 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--trt-min-shapes 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--trt-opt-shapes 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--trt-max-shapes 'x:[1,4,32,48]' 'timesteps:[1]' 'context:[1,77,768]' 'control_0:[1,320,32,48]' 'control_1:[1,320,32,48]' 'control_2:[1,320,32,48]' 'control_3:[1,320,16,24]' 'control_4:[1,640,16,24]' 'control_5:[1,640,16,24]' 'control_6:[1,640,8,12]' 'control_7:[1,1280,8,12]' 'control_8:[1,1280,8,12]' 'control_9:[1,1280,4,6]' 'control_10:[1,1280,4,6]' 'control_11:[1,1280,4,6]' 'control_12:[1,1280,4,6]' \
--load-inputs onnx_input.json --load-outputs onnx_res.json \
--trt-outputs mark all \
--abs 1e-3 --rel 1e-3 

polygraphy debug precision ../LeViT-128S.onnx \
-v --int8 --workspace 28G --no-remove-intermediate --log-file ./log_file.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
-p fp16 --mode bisect --dir forward --show-output --calibration-cache ./cal_trt.bin \
--artifacts ./polygraphy_debug.engine --art-dir ./art-dir \
--check \
polygraphy run polygraphy_debug.engine \
--trt --load-outputs ../fp16/onnx_res.json --load-inputs ../fp16/onnx_input.json \
--abs 1e-2 -v --rel 1e-2