# controlnet-stable_diffusion-tensorrt
## 准备工作
1、安装nvidia-docker \
2、拉取docker镜像
```
docker pull registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2
```
3、模型下载（步骤2中镜像包含.pth模型） \
[百度网盘链接](https://pan.baidu.com/s/1FVk1wYBX32gosUxopEdBbw?pwd=uxmx)
## 项目部署
```
# 运行容器
docker run --gpus all \
  --name trt2023 \
  -u root \
  -d \
  --ipc=host \
  --ulimit memlock=-1 \
  --restart=always \
  --ulimit stack=67108864 \
  -v ${PWD}:/home/player/ControlNet/ \
  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2 sleep 8640000
```

## 测试运行
```
chomod  +x  preprocess.sh && ./preprocess.sh
python3 compute_score.py
```
