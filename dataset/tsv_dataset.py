from tkinter.messagebox import NO
import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
from .base_dataset_GLIGEN import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid  
from io import BytesIO
import random

from .tsv import TSVFile

from io import BytesIO
import base64
from PIL import Image
import numpy as np
import albumentations as A

import sys
sys.path.append("../datasets")
from datasets.data_utils import *



# borrow from GLIGEN
def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr

def decode_item(item):
    item = json.loads(item)
    item['image'] = decode_base64_to_pillow(item['image'])

    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item

def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field

def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        data_info.pop("sentence_id", None)  # sentence id for each image (multiple sentences for one image)
        data_info.pop("dataset_name", None)  
        data_info.pop("data_source", None) 
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None) # I have checked that all 0 for flickr, vg, coco
        anno_info.pop("category_id", None)  # I have checked that all 1 for flickr vg. This is not always 1 for coco, but I do not think we need this annotation
        anno_info.pop("area", None)
        # anno_info.pop("id", None)
        anno_info["data_id"] = anno_info.pop("image_id")


def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [ x0, y0, x0+w, y0+h ]


def make_a_sentence(obj_names, clean=False):

    if clean:
        obj_names = [ name[:-6] if ("-other" in name) else name for name in obj_names]

    caption = ""
    tokens_positive = []
    for obj_name in obj_names:
        start_len = len(caption)
        caption += obj_name
        end_len = len(caption)
        caption += ", "
        tokens_positive.append(
            [[start_len, end_len]] # in real caption, positive tokens can be disjoint, thus using list of list
        )
    caption = caption[:-2] # remove last ", "

    return caption #, tokens_positive


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    return image_masks, text_masks





def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim).  
    this function will return the CLIP penultimate feature. 
    
    Note: to make sure getting the correct penultimate feature, the input y should not be normalized. 
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.   
    """
    return y@torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)




class TSVDataset(BaseDataset):
    def __init__(self, 
                tsv_path,
                which_layer_text='before', 
                which_layer_image="after_reproject",
                prob_use_caption=1,
                random_drop_embedding='none',
                image_size=512, 
                min_box_size=0.01,
                max_boxes_per_data=8,
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                ref_zero = False,  # ref image 用0代替
                ):
        super().__init__(random_crop, random_flip, image_size)
        self.tsv_path = tsv_path
        self.which_layer_text  = which_layer_text
        self.which_layer_image = which_layer_image
        self.prob_use_caption = prob_use_caption
        self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images
        self.ref_zero = ref_zero

        assert which_layer_text in ['before','after']
        assert which_layer_image in ['after', 'after_renorm', 'after_reproject']
        assert random_drop_embedding in ['none', 'both', 'image']
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # Last linear layer used in CLIP text encoder. Here we use it to map CLIP image embedding into penultimate text space. See Appendix in paper. 
        self.projection_matrix = torch.load('projection_matrix')

        # Load tsv data
        self.tsv_file = TSVFile(self.tsv_path)
        
        # preprocessed CLIP feature embedding length: 768  
        self.embedding_len = 768
        self.dynamic = 2
        self.gt_transforms = A.Compose(
                [
                # A.RandomResizedCrop(self.size, self.size, scale=(0.9, 1.0)),
                # A.SmallestMaxSize(max_size=size, p=1),
                # A.Resize(size, size),
                A.HorizontalFlip(),
                ],
            )

        self.cond_transforms = A.Compose(
                [
                # A.RandomResizedCrop(self.size, self.size, scale=(0.9, 1.0)),
                # A.SmallestMaxSize(max_size=size, p=1),
                # A.CenterCrop(size, size),
                
                A.Resize(224, 224),   #用中间这三个
                # A.HorizontalFlip(),
                # A.Rotate(limit=20),
                
                # A.Blur(p=0.3),
                # A.ElasticTransform(p=0.3),
                ]
            )

    def total_images(self):
        return len(self)

    def get_item_from_tsv(self, index):
        # import pdb; pdb.set_trace()
        _, item = self.tsv_file[index] # data id, image
        item = decode_item(item)
        return item

    def mapping(self, image_embedding):
        if self.which_layer_image == 'after':
            # use CLIP image feaure, the aligned feature space with norm=1. 
            return image_embedding
        elif self.which_layer_image == 'after_renorm':
            # same as before but normalize it to 28.7, which is empirically same as text penultimate feature norm.
            return image_embedding*28.7
        elif self.which_layer_image == 'after_reproject':
            # Re-project the CLIP image feature into text penultimate space using text linear matrix and norm it into 28.7
            image_embedding = project( image_embedding.unsqueeze(0), self.projection_matrix.T )
            image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding / image_embedding.norm() 
            image_embedding = image_embedding * 28.7 
            return image_embedding

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def draw_bboxes(self, tensor, boxes, color=(1.0, 1.0, 1.0), thickness=2):
        """
        在PyTorch图像张量上绘制边界框边框，针对[H, W, C]的张量布局。

        参数:
        - tensor: 图像张量，形状为(H, W, C)，H为高度，W为宽度，C为通道数。
        - boxes: 边界框列表，每个边界框的格式为[x_min, y_min, x_max, y_max]，坐标为归一化值。
        - color: 边框的颜色，默认为白色。
        - thickness: 边框的厚度。
        """
        H, W, C = tensor.shape
        # import pdb; pdb.set_trace()
        
        x_min, y_min, x_max, y_max = boxes
        x_min, y_min, x_max, y_max = int(x_min * W), int(y_min * H), int(x_max * W), int(y_max * H)

        # 绘制顶部和底部
        tensor[y_min:y_min+thickness, x_min:x_max, :] = torch.tensor(color)
        tensor[y_max-thickness:y_max, x_min:x_max, :] = torch.tensor(color)

        # 绘制左侧和右侧
        tensor[y_min:y_max, x_min:x_min+thickness, :] = torch.tensor(color)
        tensor[y_min:y_max, x_max-thickness:x_max, :] = torch.tensor(color)

        return tensor

    def __getitem__(self, index):
        
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        raw_item = self.get_item_from_tsv(index)
        is_det = raw_item.get('is_det', False) # if it is from detection (such as o365), then we will make a pseudo caption

        out = {}
        out['path'] =raw_item['file_name']
        # import pdb; pdb.set_trace()
        # out['keys'] =raw_item.keys()
        # -------------------- id and image ------------------- # 
        out['id'] = raw_item['data_id']
        image = raw_item['image']
        image_tensor, trans_info = self.transform_image(image)
        out["jpg"] = image_tensor  #[3,512,512]到【512,512,3】

        # -------------------- grounding token ------------------- # 
        annos = raw_item['annos']  # len: 7
        
        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_image_embeddings = []
        all_ref_images = []
        if is_det:
            all_category_names = []

        text_embedding_name = 'text_embedding_before' if self.which_layer_text == 'before' else 'text_embedding_after'
        image_embedding_name = 'image_embedding_after'
        
        for anno in annos:
            x, y, w, h = anno['bbox']
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)
            # anno.keys(): 'category_id', 'id', 'bbox', 'tokens_positive', 'data_id', 
            # 'text_embedding_before', 'text_embedding_after', 'image_embedding_before', 'image_embedding_after'
            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_masks.append(1)
                
                bbox = torch.tensor([x0,y0,x1,y1])   #/ self.image_size
                # import pdb; pdb.set_trace()
                all_ref_images.append(image_tensor[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

                all_text_embeddings.append(anno[text_embedding_name])
                all_image_embeddings.append(  self.mapping(anno[image_embedding_name])  )
                if is_det:
                    all_category_names.append(anno["category_name"])
        # import pdb; pdb.set_trace()
        #这块没有问题
        
        
        # len(all_boxes:6)
        # Sort according to area and choose the largest N objects   
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data, 4)  # [30,4]
        masks = torch.zeros(self.max_boxes_per_data)
        text_embeddings =  torch.zeros(self.max_boxes_per_data, self.embedding_len)
        image_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        if is_det:
            category_names = []

        # add layout image
        # 创建一个全黑的图片
        layout_all = []
        layout_multi = torch.zeros(image_tensor.shape[1], image_tensor.shape[2], 3)
        # layout[int(boxes[i][1]*self.image_size):int(boxes[i][3]*self.image_size), int(boxes[i][0]*self.image_size):int(boxes[i][2]*self.image_size)] = [1.0, 1.0, 1.0]
        
        if len(all_ref_images) != 0:
            ref_idx = wanted_idxs[0]
            all_ref_images[ref_idx] = (all_ref_images[ref_idx] + 1.0) / 2
            
            all_ref_images[ref_idx] = all_ref_images[ref_idx].permute(1,2,0)
            all_ref_images[ref_idx] = pad_to_square(all_ref_images[ref_idx], pad_value = 0, random = False)
            try:
                if not self.ref_zero:
                    out['ref'] = torch.from_numpy(self.cond_transforms(image=all_ref_images[ref_idx])['image']).permute(2,0,1)
                else:
                    out['ref'] = torch.zeros([3, 224, 224])
            except:
                out['ref'] = torch.zeros([3, 224, 224])
            # print(out['ref'].shape)
        else:
            out['ref'] = torch.zeros([3, 224, 224])
            # print(out['ref'].shape)

        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            # add layout
            # import pdb; pdb.set_trace()
            layout = np.zeros((image_tensor.shape[1], image_tensor.shape[2], 3), dtype=np.float32)
            layout[int(boxes[i][1]*self.image_size):int(boxes[i][3]*self.image_size), int(boxes[i][0]*self.image_size):int(boxes[i][2]*self.image_size)] = [1.0, 1.0, 1.0]
            layout_all.append(layout)

            layout_multi = self.draw_bboxes(layout_multi, boxes[i])

            text_embeddings[i] =  all_text_embeddings[idx]
            image_embeddings[i] = all_image_embeddings[idx]
            if is_det:
                category_names.append(all_category_names[idx])

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            image_masks = masks
            text_masks = masks
        
        # import pdb; pdb.set_trace()
        
        out["box_ref"] = boxes[0].unsqueeze(0)
        out["image_embeddings_ref"] = image_embeddings[0].unsqueeze(0)
        out["image_mask_ref"] = image_masks[0].unsqueeze(0)
        out["text_mask_ref"] = text_masks[0].unsqueeze(0)
        
        # print(layout_all[0].shape)
        
        #这块有问题
        # import pdb; pdb.set_trace()
        if len(layout_all) != 0:
            out['layout'] = torch.from_numpy(layout_all[0])
            out['layout_all'] = layout_multi
            # print(out['layout'].shape)
        else:
            out['layout'] = torch.zeros([512,512,3])
            out['layout_all'] = torch.zeros([512,512,3])

            
        # 做hint
        
        # for i, idx in enumerate(wanted_idxs):
        #     boxes[i] = all_boxes[idx]
        #     masks[i] = all_masks[idx]
        # ref_box_yyxx = boxes[0]
        # # ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        # masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        
        # y1,y2,x1,x2 = ref_box_yyxx
        # ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        # ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        # ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
        
        # collage = np.zeros(cropped_target_image.shape, dtype=np.uint8) #改成纯黑的背景
        # # collage = cropped_target_image.copy() 
        # collage[y1:y2,x1:x2,:] = ref_image_collage
        # H1, W1 = collage.shape[0], collage.shape[1]
        
        # cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        # collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        # collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        # H2, W2 = collage.shape[0], collage.shape[1]
        # # import pdb; pdb.set_trace()
        # cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        # collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
        # collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
        # collage_mask[collage_mask == 2] = -1
        
        # # Prepairing dataloader items
        # masked_ref_image_aug = masked_ref_image_aug  / 255 
        # cropped_target_image = cropped_target_image / 127.5 - 1.0
        # collage = collage / 127.5 - 1.0 
        # collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
        
        
        
        
           
        out["boxes"] = boxes
        # print(boxes.shape)
        out["masks"] = masks # indicating how many valid objects for this image-text data
        out["image_masks"] = image_masks # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks # indicating how many objects still there after random dropping applied
        out["text_embeddings"] =  text_embeddings    # [30,768]
        out["image_embeddings"] = image_embeddings   # [30,768]   
        # add other things
        out["time_steps"] = self.sample_timestep()
        
        
        # import pdb; pdb.set_trace()
        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            if is_det:
                out["caption"] = make_a_sentence(category_names)
            else:
                out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""
        out['txt'] = raw_item["caption"]
        
        return out

    def __len__(self):
        return len(self.tsv_file)


