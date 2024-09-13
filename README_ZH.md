[English](./README.md)| [中文](./README_ZH.md)


# 使用Qdrant + CNCLIP + Gradio 实现图文检索

![Alt text](./data/assets/image1.png)

![Alt text](./data/assets/image2.png)


## 1、数据准备

下载链接：[图文检索图片数据](https://tianchi.aliyun.com/competition/entrance/532031/information)

```python
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
import os

data_path = "./data/MR_valid_imgs.tsv"
save_dir = "./data/images"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = pd.read_csv(data_path, header=None, sep='\t')

for index, row in data.iterrows():
    image_id = row[0]
    image_data = row[1]

    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_data)))
    img.save(os.path.join(save_dir, f"{image_id}.png"))

```

利用以上代码将图片和图片base64数据保存为本地png图片，放到：`data/images` 文件夹下


## 2、 安装依赖

(1) 首先安装cn_clip

因为官方仓库有一点小bug，所以从我fork后的仓库安装
```bash
git clone https://github.com/seanzhang-zhichen/Chinese-CLIP.git
cd Chinese-CLIP
pip install -e .
```

**注意：** 必须得从源码安装，否则会报错缺少配置文件


(2) 安装其它依赖

```bash
pip install -r requirements.txt
```


## 3、启动web demo

```bash
python web_demo.py
```


## 其它

如果想使用 tensorrt 推理，安装 tensorrt==8.6.1 版本即可（已测试），会比torch快很多

