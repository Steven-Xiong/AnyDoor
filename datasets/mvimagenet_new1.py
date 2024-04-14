import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset,BaseDataset_t2i
import csv
import glob
import matplotlib.pyplot as plt
from datasets.data_transfer import prepare_batch_hetero
from transformers import CLIPProcessor, CLIPModel
from datasets.data_transfer import prepare_batch_hetero, get_clip_feature,project

from torchvision import transforms
from PIL import Image

class MVImageNetDataset(BaseDataset_t2i):
    def __init__(self, txt, image_dir):
        with open(txt,"r") as f:
            data = f.read().split('\n')[:-1]    
        self.image_dir = image_dir 
        self.data = data
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2
        self.mask_dir = image_dir + 'mask/'
        self.caption_dir = image_dir + 'captions/all.csv'
        
        # load captions
        self.caption_index = {}
        with open(self.caption_dir, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                path, caption = os.path.join(row[0].split('/')[-4],row[0].split('/')[-3],row[0].split('/')[-2],row[0].split('/')[-1]), row[1]
                self.caption_index[path] = caption
        # image_count = 0
        # for folder in self.data:
        #     image_count += len(glob.glob(folder.replace('MVDir/', self.image_dir)))
        # import pdb; pdb.set_trace()
        
        version = "openai/clip-vit-large-patch14"
        self.clip_model = CLIPModel.from_pretrained(version) #.cuda()
        self.processor = CLIPProcessor.from_pretrained(version)
        self.transform_to_pil = transforms.ToPILImage()
        # import pdb; pdb.set_trace()
        self.get_clip_feature = get_clip_feature(model=self.clip_model, processor=self.processor,input=None, is_image=True)
        self.projection_matrix = torch.load('projection_matrix') #.cuda()
        self.max_boxes = 8
        
    def __len__(self):
        image_count = 0
        for folder in self.data:
            image_count += len(glob.glob(folder.replace('MVDir/', self.image_dir)))
        return image_count

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag

    def get_alpha_mask(self, mask_path):
        # import pdb; pdb.set_trace()
        image = cv2.imread( mask_path) #, cv2.IMREAD_UNCHANGED
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        return mask
    
    def seg2bbox(self, mask, ratio=0.1):  #borrow from instantbooth
        mask = mask[:, :, 0]
        h, w= mask.shape[0], mask.shape[1]
        # crop exact bbox from mask. 
        y, x = np.where(mask!=0)
        xmin, xmax = np.min(x), np.max(x)+1
        ymin, ymax = np.min(y), np.max(y)+1
        tight_crop = (ymin, xmin, ymax, xmax)

        # expand according to ratio.
        ybox, xbox = ymax - ymin, xmax - xmin
        ycenter, xcenter = (ymin + ymax) // 2, (xmin + xmax) // 2
        ratio = min(ratio, h*1./ybox - 1., w*1./xbox - 1.)
        ynew, xnew = int(ybox * (1+ratio)), int(xbox * (1+ratio))
        ymin_new, ymax_new = max(ycenter - ynew//2, 0), min(ycenter + ynew//2, h)
        xmin_new, xmax_new = max(xcenter - xnew//2, 0), min(xcenter + xnew//2, w)
        expanded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)
        
        # pad to square bbox.
        ybox_new, xbox_new = ymax_new-ymin_new, xmax_new - xmin_new
        if xbox_new < ybox_new:
            pad = (ybox_new )//2
            xmin_new, xmax_new = max(xcenter - pad, 0), min(xcenter + pad, w)
        else:
            pad = (xbox_new )//2
            ymin_new, ymax_new = max(ycenter - pad, 0), min(ycenter + pad, h)
        # cropped_mask_pad = mask[ymin_new:ymax_new, xmin_new:xmax_new]
        # cv2.imwrite('cropped_mask_pad.png', cropped_mask_pad)
        padded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)

        return padded_crop
    
    # def project(self, x, projection_matrix):
    #     """
    #     x (Batch*768) should be the penultimate feature of CLIP (before projection)
    #     projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    #     defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    #     this function will return the CLIP feature (without normalziation)
    #     """
    #     return x@torch.transpose(projection_matrix, 0, 1)

    # def get_clip_feature(self, model, processor, input, is_image=False):
    #     which_layer_text = 'before'
    #     which_layer_image = 'after_reproject'
        
    #     if is_image:
    #         if isinstance(input, list):
    #             if None in input: return None
    #         else:
    #             if input == None: return None
    #         # import pdb; pdb.set_trace()    
    #         transform_to_pil = transforms.ToPILImage()
    #         image = transform_to_pil(input).convert("RGB")
            
    #         # image = Image.open(input).convert("RGB")
    #         inputs = processor(images=[image],  return_tensors="pt", padding=True)
    #         inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
    #         inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
    #         outputs = model(**inputs)
    #         feature = outputs.image_embeds 
    #         if which_layer_image == 'after_reproject':
    #             feature = self.project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
    #             feature = ( feature / feature.norm() )  * 28.7 
    #             feature = feature.unsqueeze(0)
    #     else:
    #         if isinstance(input, list):
    #             if None in input: return None
    #         else:
    #             if input == None: return None
    #         inputs = processor(text=input,  return_tensors="pt", padding=True)
    #         inputs['input_ids'] = inputs['input_ids'].cuda()
    #         inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
    #         inputs['attention_mask'] = inputs['attention_mask'].cuda()
    #         outputs = model(**inputs)
    #         if which_layer_text == 'before':
    #             feature = outputs.text_model_output.pooler_output
    #     return feature

    def get_sample(self, idx):
        # import pdb; pdb.set_trace()
        object_dir = self.data[idx].replace('MVDir/', self.image_dir)
        # 手动加mask dir
        mask_dir = self.data[idx].replace('MVDir/', self.mask_dir).replace('/images','')
    
        frames = os.listdir(object_dir)
        frames = [ i for i in frames ] #if '.png' in i

        # Sampling frames
        min_interval = len(frames)  // 8
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index )
        end_frame_index = min(end_frame_index, len(frames) - 1)

        # Get image path
        ref_mask_name = frames[start_frame_index] + '.png'
        tar_mask_name = frames[end_frame_index] + '.png'  #最后一帧做target

        ref_image_name = frames[start_frame_index] # ref_mask_name.split('_')[0] #+ '.jpg'
        tar_image_name = frames[end_frame_index] # tar_mask_name.split('_')[0] #+ '.jpg'

        ref_mask_path = os.path.join(mask_dir, ref_mask_name)
        tar_mask_path = os.path.join(mask_dir, tar_mask_name)
        ref_image_path = os.path.join(object_dir, ref_image_name)
        tar_image_path = os.path.join(object_dir, tar_image_name) 

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path).astype(np.uint8)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        # # add resize
        # ref_image = cv2.resize(ref_image, (512,512))

        tar_image = cv2.imread(tar_image_path).astype(np.uint8)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = self.get_alpha_mask(ref_mask_path)
        tar_mask = self.get_alpha_mask(tar_mask_path)
        
        # import pdb; pdb.set_trace()
        target_index = os.path.join(tar_image_path.split('/')[-4],tar_image_path.split('/')[-3],tar_image_path.split('/')[-2],tar_image_path.split('/')[-1])
        caption = self.caption_index[target_index]
        # import pdb; pdb.set_trace()
        bbox = self.seg2bbox(np.stack([tar_mask,tar_mask,tar_mask],-1))  #将seg map补全成mask
        # 创建一个全黑的图片
        
        # print('bbox:',bbox)
        # # 已知的bbox坐标：(ymin, xmin, ymax, xmax)
        # padded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)  # 请替换为实际的坐标值
        # 在bbox区域内填充为白色（或其他颜色）
        # 注意：颜色设置为(255, 255, 255)是白色，(R, G, B)格式

        layout = np.zeros((tar_image.shape[0], tar_image.shape[1], 3), dtype=np.uint8)
        layout[bbox[0]:bbox[2], bbox[1]:bbox[3]] = [255, 255, 255]


        item_with_collage = self.process_pairs_customized_mvimagenet(ref_image, ref_mask, tar_image, tar_mask,layout) #全部[1920,1080]      
        # import pdb; pdb.set_trace()
        # ref_image_resized = pad_to_square(ref_image, pad_value = 255, random = False)
        # ref_image_resized = cv2.resize(ref_image_resized.astype(np.uint8), (224,224)).astype(np.uint8) / 255
        # plt.imsave('ref_img_resized.jpg',item_with_collage['ref'])
        # item_with_collage['ref'] = ref_image_resized  #这样不对。应该剪裁居中并抠图，放进han'hu
        item_with_collage['anno_id'] = target_index
        item_with_collage['txt'] = caption
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps
        
        # add image embeddings和text embeddings
        # import pdb; pdb.set_trace()
        which_layer_text = 'before'
        which_layer_image = 'after_reproject'
        input=torch.from_numpy(item_with_collage['ref']).permute(2,0,1)
        if isinstance(input, list):
            if None in input: return None
        else:
            if input == None: return None  
        image = self.transform_to_pil(input).convert("RGB")
        
        # image = Image.open(input).convert("RGB")
        inputs = self.processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'] #.cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]) #.cuda()  # placeholder
        outputs = self.clip_model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, self.projection_matrix.T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            ref_embedding= feature.unsqueeze(0)
        
        item_with_collage['image_embeddings'] = np.concatenate((ref_embedding.cpu().detach().numpy(), np.zeros((self.max_boxes-1, 768))), axis=0)
        
        
        # ref_embedding = self.get_clip_feature(input=torch.from_numpy(item_with_collage['ref']).permute(2,0,1), is_image=True).cpu().detach().numpy()
        # item_with_collage['image_embeddings'] = np.concatenate((ref_embedding, np.zeros((29, 768))), axis=0)
        
        # add new keys
        item_with_collage['text_embeddings'] = np.zeros((self.max_boxes,768))
        #除了第一个其他都是0？
        array = np.zeros(self.max_boxes,dtype=np.int32)
        array[0] = 1
        item_with_collage['masks'] = array #.reshape(30,1)
        # nimport pdb; pdb.set_trace()
        # bbox需要归一化
        bbox = np.array(bbox,dtype=np.float32)
        bbox[0],bbox[1],bbox[2], bbox[3] = bbox[0]/tar_image.shape[0], bbox[1]/tar_image.shape[1], bbox[2]/tar_image.shape[0],bbox[3]/tar_image.shape[1]
        item_with_collage['boxes'] = np.concatenate((bbox.reshape(1,4), np.zeros((self.max_boxes-1,4))), axis=0) 
        item_with_collage['layout_all'] = item_with_collage['layout']
        
        return item_with_collage
