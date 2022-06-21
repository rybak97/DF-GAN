


import os
import sys

data_path = '../src/data/'
encoder_weights_path = "../text_encoder_weights/text_encoder200.pth"
image_save_path = "../gen_images"
gen_path_save = "../gen_weights"



import numpy as np
import torch

from sample import sample, save_image
from generator.model import Generator
from text_encoder.model import RNNEncoder
from utils import create_loader




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


generator = Generator(n_channels=32, latent_dim=100).to(device)
generator.load_state_dict(torch.load("../gen_weights/gen_epoch_310.pth", map_location=device))
generator = generator.eval()


# In[6]:


train_loader = create_loader(256, 24, "../data", "test")


# In[7]:


n_words = train_loader.dataset.n_words


# In[8]:


text_encoder = RNNEncoder.load("../text_encoder_weights/text_encoder200.pth", n_words)
text_encoder.to(device)

for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder = text_encoder.eval()


# In[10]:


get_ipython().run_cell_magic('time', '', 'sample(generator, text_encoder, next(iter(train_loader)), "../gen_images")')


# # Own birds

# In[11]:


dataset = train_loader.dataset


# In[13]:


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
    img =  generator(noise, embed)
    save_image(img[0].data.cpu().numpy(), "../gen_images", name)


# In[22]:


caption = "this green small bird"
gen_own_bird(caption, caption)

if __name__ == '__main__':
    gen_own_bird(caption, caption)
