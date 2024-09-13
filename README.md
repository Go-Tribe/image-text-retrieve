# Using Qdrant + CNCLIP + Gradio to Implement Image-Text Retrieval

![Alt text](./data/assets/image1.png)

![Alt text](./data/assets/image2.png)


## 1. Data Preparation

Download link: [Image-Text Retrieval Image Data](https://tianchi.aliyun.com/competition/entrance/532031/information)

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

The above code saves the images and their base64 data as local PNG images in the `data/images` folder.


## 2. Install Dependencies

(1) First, install cn_clip

Since there is a small bug in the official repository, install it from my forked repository:
```bash
git clone https://github.com/seanzhang-zhichen/Chinese-CLIP.git
cd Chinese-CLIP
pip install -e .
```

**Note:** You must install from the source code, otherwise you will get an error about missing configuration files.

(2) Install other dependencies
```bash
pip install -r requirements.txt
```


## 3. Start the Web Demo

```bash
python web_demo.py
```



## Others

If you want to use TensorRT inference, install the `tensorrt==8.6.1` version (already tested), which will be much faster than PyTorch.
