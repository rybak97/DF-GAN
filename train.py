import os
from typing import List, Tuple


import numpy as np
import torch

from sample import sample, save_image
from generator.model import Generator
from text_encoder.model import RNNEncoder
from utils import create_loader

from deep_fusion_gan.model import DeepFusionGAN
from utils import create_loader, fix_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = create_loader(256, 24, "../src/data/", "test")
n_words = train_loader.dataset.n_words

text_encoder = RNNEncoder.load("../text_encoder_weights/text_encoder200.pth", n_words)
text_encoder.to(device)

for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder = text_encoder.eval()

dataset = train_loader.dataset

generator = Generator(n_channels=32, latent_dim=100).to(device)
generator.load_state_dict(torch.load("../gen_weights/gen_39.pth", map_location=device))
generator = generator.eval()

def gen_own_bird(word_caption, name):
    codes = [dataset.word2code[w] for w in word_caption.lower().split()]

    caption = np.array(codes)
    pad_caption = np.zeros((18, 1), dtype='int64')

    if len(caption) <= 18:
        pad_caption[:len(caption), 0] = caption
        len_ = len(caption)
    else:
        indices = list(np.arange(len(caption)))
        np.random.shuffle(indices)
        pad_caption[:, 0] = caption[np.sort(indices[:18])]
        len_ = 18

    embed = text_encoder(torch.tensor(pad_caption).reshape(1, -1), torch.tensor([len_]))
    batch_size = embed.shape[0]
    noise = torch.randn(batch_size, 100).to(device)
    img = generator(noise, embed)
    save_image(img[0].data.cpu().numpy(), "../gen_images", name)


def train() -> Tuple[List[float], List[float], List[float]]:
    fix_seed()

    data_path = '../src/data/'
    encoder_weights_path = "../text_encoder_weights/text_encoder200.pth"
    image_save_path = "../gen_images"
    gen_path_save = "../gen_weights"

    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(gen_path_save, exist_ok=True)

    train_loader = create_loader(256, 24, data_path, "train")
    model = DeepFusionGAN(n_words=train_loader.dataset.n_words,
                          encoder_weights_path=encoder_weights_path,
                          image_save_path=image_save_path,
                          gen_path_save=gen_path_save)

    return model.fit(train_loader)

caption = "black bird with green beak"
gen_own_bird(caption, caption)


if __name__ == '__main__':
    gen_own_bird(caption, caption)
