# 使用Qdrant + cnclip + gradio 实现图文检索


## 1. 安装依赖

1、首先安装cn_clip

因为官方仓库有一点小bug，所以从我fork后的仓库安装
```bash
git clone https://github.com/seanzhang-zhichen/Chinese-CLIP.git
cd Chinese-CLIP
pip install -e .
```

**注意：** 必须得从源码安装，否则会报错缺少配置文件


2、安装其它依赖

```bash
pip install -r requirements.txt
```


## 2、启动web demo

```bash
python web_demo.py
```


## 其它

如果想使用 tensorrt 推理，安装 tensorrt==8.6.1 版本即可（已测试），会比torch快很多