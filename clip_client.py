import torch
import cn_clip.clip as clip
from PIL import Image

class ChineseClipTorch:
    def __init__(self, model_name="ViT-L-14", device="cuda", download_root='./data/model'):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load_from_name(model_name, device=self.device, download_root=download_root)
        self.model.eval()

    def compute_image_features(self, image_path):
        image = Image.open(image_path)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.tolist()

    def compute_text_features(self, texts):
        text = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.tolist()


if __name__ == "__main__":
    image_path = "./data/images/130.png"
    clip_client = ChineseClipTorch()
    image_embedding = clip_client.compute_image_features(image_path)
    print(image_embedding)

    text = "皮卡丘"
    text_embedding = clip_client.compute_text_features(text)
    print(text_embedding)