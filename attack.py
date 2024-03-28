import numpy as np
import torch
import random

from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer
from utils import search_min_sentence_iteration, genetic, PGDattack, get_char_table, train,image_grid
from modelscope import snapshot_download

# ==========================加载模型=============================
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
device2 = torch.device('cuda:1')
len_prompt = 5


model_dir = snapshot_download('xiaoguailin/clip-vit-large-patch14')
tokenizer = CLIPTokenizer.from_pretrained(model_dir)
text_encoder = CLIPTextModel.from_pretrained(model_dir)
text_encoder = nn.DataParallel(text_encoder, device_ids=[0, 1])

text_encoder = text_encoder.to(device)
char_table = get_char_table()

# ==========================生成想要攻击的句子=============================
attack_sentence = "a car and a woman"

#  Greedy
greedy_sentence = search_min_sentence_iteration(attack_sentence, char_table, len_prompt,
                                                1, tokenizer=tokenizer, text_encoder=text_encoder)
print("greedy_prompt:", greedy_sentence)

#  Genetic
for i in range(5):
    genetic_prompt = genetic(attack_sentence, char_table, len_prompt, tokenizer=tokenizer,
                             text_encoder=text_encoder)
    genetic_sentence = attack_sentence + ' ' + genetic_prompt[0][0]
    print("genetic_prompt:", genetic_sentence)

#  PGD
for i in range(5):
    max_tensor, loss_list, pgd_prompt, max_loss = train(init_per_sample=1, sentence=attack_sentence,
                                                        len_prompt=len_prompt, char_list=char_table,
                                                        model=text_encoder.text_model, iter_num=100,
                                                        eta_u=1, tokenizer=tokenizer, text_encoder=text_encoder)
    pgd_sentence = attack_sentence + ' ' + pgd_prompt
    print("pgd_prompt:", pgd_sentence)

# ==========================Targeted QF Attack  Step 1: Generate mask=============================
from utils import object_key
sentence_list = [
    "A black panther lying in a jungle and a women",
    "A fishing boat on a lake at sunrise and a women",
    "A tea cup on a saucer with a teapot and a women",
    "A man playing guitar on a street corner and a women",
    "A group of flamingos standing in a pond and a women",
    "A fireflies in a field at dusk and a women",
    "A train chugging through a countryside and a women",
    "A butterfly on a colorful flower and a women",
    "A soccer game being played on a stadium and a women",
    "A man kayaking down a river through rapids and a women"
]
#掩码向量用于分析句子中与目标词相关的部分。
mask = object_key(sentence_list, 'and a women', thres=9, tokenizer=tokenizer, text_encoder=text_encoder)

# ==========================Targeted QF Attack  Step 2: Generate perturbation prompts=============================
attack_sentence = "a car and a woman"
#mask 张量的形状重新调整为一维向量（flatten），即将多维的张量转换为一维的张量。将张量重新调整为一维向量的操作可以使得数据在某些情况下更容易处理
mask = mask.view(-1)

#  Greedy
greedy_sentence = search_min_sentence_iteration(attack_sentence, char_table, len_prompt,
                                                1, tokenizer=tokenizer, text_encoder=text_encoder, mask=mask)
print("greedy_prompt:", greedy_sentence)

#  Genetic
for i in range(10):
    genetic_prompt = genetic(attack_sentence, char_table, len_prompt, tokenizer=tokenizer,
                             text_encoder=text_encoder, mask=mask)
    genetic_sentence = attack_sentence + ' ' + genetic_prompt[0][0]
    print("genetic_prompt:", genetic_sentence)

#  PGD
for i in range(10):
    max_tensor, loss_list, pgd_prompt, max_loss = train(init_per_sample=1, sentence=attack_sentence,
                                                        len_prompt=len_prompt, char_list=char_table,
                                                        model=text_encoder.text_model, iter_num=100,
                                                        eta_u=1, tokenizer=tokenizer, text_encoder=text_encoder,
                                                        mask=mask)
    pgd_sentence = attack_sentence + ' ' + pgd_prompt
    print("pgd_prompt:", pgd_sentence)

# ==========================Evaluation=============================
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained(
    '/home/gaozx/.cache/modelscope/hub/AI-ModelScope/stable-diffusion-v1-4', revision='fp16',
    torch_dtype=torch.float16)
pipe = pipe.to(device)

# ==========================Generate Images and Show Results=============================
from torch import autocast
generator = torch.Generator("cuda").manual_seed(27)

original_sentence = 'a car and a woman'
perturbation_prompt = '-08=*'
sentence = original_sentence + ' ' + perturbation_prompt

num_images = 4
prompt = [sentence] * num_images
with autocast('cuda'):
    images = pipe(prompt, generator=generator, num_inference_steps=50).images
    # torch.cuda.empty_cache()  # 释放PyTorch缓存占用的内存

grid = image_grid(images, rows=1, cols=4)
grid