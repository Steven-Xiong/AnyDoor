import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'),strict = False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)



def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    import pdb; pdb.set_trace()
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image


def inference_single_image_new(item, guidance_scale = 5.0):
    # item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    # import pdb; pdb.set_trace()
    ref = item['ref'][0].numpy() * 255
    tar = item['jpg'][0].numpy() * 127.5 + 127.5
    hint = item['hint'][0].numpy() * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][0][:,:,-1].numpy() * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref'][0].numpy()
    tar = item['jpg'][0].numpy()
    hint = item['hint'][0].numpy()
    caption = item['txt'][0]
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()   #加txt

    c_image = model.get_learned_conditioning( clip_input )
    c_txt = model.get_learned_conditioning_txt(caption)
    c = torch.cat((c_image,c_txt),dim=1) 

    guess_mode = False
    H,W = 512,512
    
    
    cond = {"c_concat": [control], "c_crossattn": [c]}

    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_unconditional_conditioning(num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes'][0]
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'][0] 

    gen_image = pred
    # import pdb; pdb.set_trace()
    # gen_image = crop_back(pred, tar, sizes, tar_box_yyxx_crop) 
    return gen_image

import clip

# clip socre计算
# def calculate_clip_score(dataloader, model, real_flag, fake_flag):
#     score_acc = 0.
#     sample_num = 0.
#     logit_scale = model.logit_scale.exp()
#     for batch_data in tqdm(dataloader):
#         real = batch_data['real']
#         real_features = forward_modality(model, real, real_flag)
#         fake = batch_data['fake']
#         fake_features = forward_modality(model, fake, fake_flag)
        
#         # normalize features
#         real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
#         fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
        
#         # calculate scores
#         # score = logit_scale * real_features @ fake_features.t()
#         # score_acc += torch.diag(score).sum()
#         score = logit_scale * (fake_features * real_features).sum()
#         score_acc += score
#         sample_num += real.shape[0]
    
#     return score_acc / sample_num


class DINOEvaluator:
    def __init__(self, device, dino_model='facebook/dino-vits16') -> None:
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

    @torch.inference_mode()
    def get_image_features(self, images) -> torch.Tensor:
        inputs = processor(images=images, return_tensors="pt").to(device=self.device)
        features = model(**inputs).last_hidden_state[:, 0, :]

    @torch.inference_mode()
    def img_to_img_similarity(self, src_images, generated_images, reduction=True):
        src_features = self.get_image_features(src_images)
        gen_features = self.get_image_features(src_images)

        return torchmetrics.functional.pairwise_cosine_similarity(src_features, gen_features).mean().item()

def calc_clip(clip, generated_images, reference_images, prompts, placeholder_token,
              step=0, split='train'):
    clip_img_score = clip.img_to_img_similarity(reference_images, generated_images).item()
    prompts = [x.replace(placeholder_token, '') for x in prompts]
    clip_txt_score = clip.txt_to_img_similarity(prompts, generated_images).item()
    logs = {f"{split}_clip_img_score": clip_img_score, f"{split}_clip_txt_score": clip_txt_score}

    return logs

def calc_dino_div(dino, div, generated_images, split="train"):
    dino_score = dino.img_to_img_similarity(reference_images, generated_images).item()
    div_score = div.get_score(generated_images).item()
    logs = {f"{split}_dino_score": dino_score, f"{split}_div_score": div_score}

    return logs

from datasets.lvis_new import LvisDataset,LvisDataset_val #, LvisDataset_customized    #这里修改
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os 

from pathlib import Path
import torchmetrics
from collections import defaultdict
from accelerate import Accelerator
from transformers import ViTImageProcessor, ViTModel
import torch
import tqdm
from torch.nn.functional import cosine_similarity
from PIL import Image

import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
# from transformers import CLIPProcessor, CLIPModel
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import os
import cv2
import timm
from torchmetrics.functional.multimodal import clip_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the CLIP model
# model_ID = "clip-vit-base-patch16"
# clipmodel = CLIPModel.from_pretrained(model_ID)

clipmodel, preprocess = clip.load("ViT-B/32", device=device)

# preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# model_name = 'dino_vits16'  # 选择一个适合的DINO模型
# dinomodel = timm.create_model(model_name, pretrained=True)
# dinomodel.eval()

# dinomodel.to(device)
dinomodel = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

from torchmetrics.multimodal.clip_score import CLIPScore  #clip-T
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")



# 加载CLIP模型
# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    # import pdb;pdb.set_trace()
    # images_int = (np.asarray(images[0]) * 255).astype("uint8")
    images_int = (np.asarray(images) * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


# CLIP-I计算标准： CLIP-I is the average pairwise cosine similarity between CLIP embeddings of generated and real images.
# CLIP-T计算标准： The second important aspect to evaluate is prompt fidelity, measured as the average cosine similarity between prompt and image CLIP embeddings. We denote this as CLIP-T
if __name__ == '__main__': 
    '''
    # ==== Example for inferring a single image ===
    reference_image_path = './examples/TestDreamBooth/FG/01.png'
    bg_image_path = './examples/TestDreamBooth/BG/000000309203_GT.png'
    bg_mask_path = './examples/TestDreamBooth/BG/000000309203_mask.png'
    save_path = './examples/TestDreamBooth/GEN/gen_res.png'

    # reference image + reference mask
    # You could use the demo of SAM to extract RGB-A image with masks
    # https://segment-anything.com/demo
    image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
    mask = (image[:,:,-1] > 128).astype(np.uint8)
    image = image[:,:,:-1]
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    ref_image = image 
    ref_mask = mask

    # background image
    back_image = cv2.imread(bg_image_path).astype(np.uint8)
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # background mask 
    tar_mask = cv2.imread(bg_mask_path)[:,:,0] > 128
    tar_mask = tar_mask.astype(np.uint8)
    
    gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
    h,w = back_image.shape[0], back_image.shape[0]
    ref_image = cv2.resize(ref_image, (w,h))
    vis_image = cv2.hconcat([ref_image, back_image, gen_image])
    
    cv2.imwrite(save_path, vis_image [:,:,::-1])
    '''
    #'''
    # ==== Example for inferring VITON-HD Test dataset ===

    
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = './Lvis_val_try1'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_dir = DConf.Test.VitonHDTest.image_dir
    image_names = os.listdir(test_dir)
    

    test_dir = DConf.Test.Lvis.image_dir
    dataset12 = LvisDataset_val(**DConf.Test.Lvis)
    image_data = [dataset12]
    dataset = ConcatDataset(image_data)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)
    
    CLIP_T = []
    CLIP_I = []
    DINO_I = []


    for i, batch in enumerate(dataloader):
        if i >5:
            break
        print(batch.keys())
        ref_image = batch['ref'][0]
        gt_image = batch['jpg'][0]
        caption = batch['txt'][0]
        
        gen_image = inference_single_image_new(batch)   #(ref_image, ref_mask, gt_image.copy(), tar_mask)
        
        # import pdb; pdb.set_trace()
        image_name = batch['img_path'][0].split('/')[-1]
        gen_path = os.path.join(save_dir, image_name)
        
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(gen_path.replace('.jpg','ref.jpg'),ref_image)
        cv2.imwrite(gen_path,gen_image)
        
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            
            # gen_image = np.transpose(np.expand_dims(gen_image, axis=0), (0, 3, 1, 2))
            gen_image = np.transpose(torch.from_numpy(gen_image.astype(np.uint8)).unsqueeze(0),(0,3,1,2)) #.transpose(0,3,1,2)
            CLIP_T.append(clip_score(gen_image, caption, "openai/clip-vit-base-patch16"))
            
    #         gt_image = np.transpose((gt_image* 127.5 + 127.5).unsqueeze(0),(0,3,1,2))
    #         gt_clip_features = clipmodel.encode_image(preprocess(gt_image))
    #         generated_clip_features = clipmodel.encode_image(preprocess(gen_image))
    #         CLIP_I.append(cosine_similarity(gt_clip_features, generated_clip_features).item())
            
    #         # CLIP_I.append(calculate_clip_score(images, prompts))
            
    #         gt_dino_features = dinomodel(gt_image)
    #         generated_dino_features = dinomodel(gen_image)
    #         DINO_I.append(cosine_similarity(gt_dino_features, generated_dino_features).item())

            
    clip_t_scores = np.mean(CLIP_T)
    # clip_i_scores = np.mean(CLIP_I)
    # dino_i_scores = np.mean(DINO_I)
    # print('clip_t_scores:',clip_t_scores,'clip_i_scores:',clip_i_scores,'dino_i_scores:',dino_i_scores)

        # vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
        # cv2.imwrite(gen_path, vis_image[:,:,::-1])
        

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,AutoProcessor, CLIPModel, AutoModel
    import torch.nn as nn
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    clipprocessor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    dinoprocessor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dinomodel = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    
    for i, batch in enumerate(dataloader):
        if i >5:
            break
        # print(batch.keys())
        ref_image = batch['ref'][0]
        gt_image = batch['jpg'][0]
        caption = batch['txt'][0]
        # gen_image = inference_single_image_new(batch)   #(ref_image, ref_mask, gt_image.copy(), tar_mask)
        
        # import pdb; pdb.set_trace()
        image_name = batch['img_path'][0].split('/')[-1]
        gen_path = os.path.join(save_dir, image_name)
        
        # cv2.imwrite(gen_path,gen_image)
        
        image1 = Image.open(batch['img_path'][0])  # ground truth
        image2 = Image.open(gen_path)              # generated image
        
        with torch.no_grad():
            inputs1 = clipprocessor(images=image1, return_tensors="pt").to(device)
            image_features1 = clipmodel.get_image_features(**inputs1)
            inputs2 = clipprocessor(images=image2, return_tensors="pt").to(device)
            image_features2 = clipmodel.get_image_features(**inputs2)
            
            cos = nn.CosineSimilarity(dim=0)
            sim = cos(image_features1[0],image_features2[0]).item()
            sim_clip_i = (sim+1)/2
            print('CLIP Similarity:', sim_clip_i)
            CLIP_I.append(sim_clip_i)
            
            
            inputs1_dino = dinoprocessor(images=image1, return_tensors="pt").to(device)
            outputs1 = dinomodel(**inputs1_dino)
            inputs2_dino = dinoprocessor(images=image2, return_tensors="pt").to(device)
            outputs2 = dinomodel(**inputs2_dino)
            
            image_features1_dino = outputs1.last_hidden_state
            image_features1_dino = image_features1_dino.mean(dim=1)
            image_features2_dino = outputs2.last_hidden_state
            image_features2_dino = image_features2_dino.mean(dim=1)
            cos = nn.CosineSimilarity(dim=0)
            sim = cos(image_features1_dino[0],image_features2_dino[0]).item()
            sim_dino = (sim+1)/2
            print('DINO Similarity:', sim_dino)
            DINO_I.append(sim_dino)
            
    clip_i_scores = np.mean(CLIP_I)
    dino_i_scores = np.mean(DINO_I) 
    print('clip_t_scores:',clip_t_scores,'clip_i_scores:',clip_i_scores,'dino_i_scores:',dino_i_scores)
        
        
        
        
        
        
        
    # #Extract features from image1
    # image1 = Image.open('img1.jpg')
    # with torch.no_grad():
    #     inputs1 = processor(images=image1, return_tensors="pt").to(device)
    #     image_features1 = model.get_image_features(**inputs1)
    # #Extract features from image2
    # image2 = Image.open('img2.jpg')
    # with torch.no_grad():
    #     inputs2 = processor(images=image2, return_tensors="pt").to(device)
    #     image_features2 = model.get_image_features(**inputs2)
    # #Compute their cosine similarity and convert it into a score between 0 and 1
    # cos = nn.CosineSimilarity(dim=0)
    # sim = cos(image_features1[0],image_features2[0]).item()
    # sim = (sim+1)/2
    # print('Similarity:', sim)


    # from transformers import AutoImageProcessor, AutoModel
    # from PIL import Image
    # import torch.nn as nn

    # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    # model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)


    # image1 = Image.open('img1.jpg')
    # with torch.no_grad():
    #     inputs1 = processor(images=image1, return_tensors="pt").to(device)
    #     outputs1 = model(**inputs1)
    #     image_features1 = outputs1.last_hidden_state
    #     image_features1 = image_features1.mean(dim=1)

    # image2 = Image.open('img2.jpg')
    # with torch.no_grad():
    #     inputs2 = processor(images=image2, return_tensors="pt").to(device)
    #     outputs2 = model(**inputs2)
    #     image_features2 = outputs2.last_hidden_state
    #     image_features2 = image_features2.mean(dim=1)

    # cos = nn.CosineSimilarity(dim=0)
    # sim = cos(image_features1[0],image_features2[0]).item()
    # sim = (sim+1)/2
    # print('Similarity:', sim)