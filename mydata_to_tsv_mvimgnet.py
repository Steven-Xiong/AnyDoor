import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
from io import BytesIO
from zipfile import ZipFile 
import multiprocessing


from tsv import TSVFile, TSVWriter

from io import BytesIO
import base64
from PIL import Image
import numpy as np
import time 
from tqdm import tqdm
from datasets.base import BaseDataset,BaseDataset_t2i

import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
# from .data_utils import * 
# from .base import BaseDataset,BaseDataset_t2i
import csv
import glob
import matplotlib.pyplot as plt

# ============= Useful fuctions and classes from Haotian =============== #
######################### COOL STUFF: NEEDED!!!! #############################


def encode_pillow_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def encode_tensor_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')

def item_to_encodable(item):
    item['image'] = encode_pillow_to_base64(item['image'])
    
    for anno in item['annos']:
        anno['text_embedding_before'] = encode_tensor_as_string(anno['text_embedding_before'])
        anno['image_embedding_before'] = encode_tensor_as_string(anno['image_embedding_before'])
        anno['text_embedding_after'] = encode_tensor_as_string(anno['text_embedding_after'])
        anno['image_embedding_after'] = encode_tensor_as_string(anno['image_embedding_after'])
    return item


######################### COOL STUFF: NEEDED!!!! #############################
# ============= Useful fuctions and classes from Haotian =============== #





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
        #anno_info.pop("category_id", None)  # I have checked that all 1 for flickr vg. This is not always 1 for coco, but I do not think we need this annotation
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


class Base():
    def __init__(self, image_root):
        self.image_root = image_root
        self.use_zip = True if image_root[-4:] == ".zip" else False 
        if self.use_zip:
            self.zip_dict = {}

    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

    def fetch_image(self, file_name):
        if self.use_zip:
            zip_file = self.fetch_zipfile(self.image_root)
            image = Image.open( BytesIO(zip_file.read(file_name)) ).convert('RGB')
        else:
            image = Image.open(  os.path.join(self.image_root,file_name)   ).convert('RGB')
        return image



class GroundingDataset_mvimgnet(BaseDataset_t2i):
    "This is for grounding data such as GoldG, SBU, CC3M, LAION"
    def __init__(self, txt, image_dir):
        # super().__init__(image_root)
        # self.image_root = image_root
        # self.json_path = json_path
        self.annotation_embedding_path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/mvimgnet_tmp' #annotation_embedding_path
        '''
        # Load raw data 
        with open(json_path, 'r') as f:
            json_raw = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        self.data = json_raw["images"] # donot name it images, which is misleading
        self.annotations = json_raw["annotations"]
      
        # clean data and annotation
        check_unique( self.data, ['id'] )
        check_unique( self.annotations, ['id'] )
        clean_data(self.data)
        clean_annotations(self.annotations)
        self.data_id_list = [  datum['data_id'] for datum in self.data   ]
        self.data = { datum['data_id']:datum  for datum in self.data } # map self.data from a list into a dict 

        # data point to its annotation mapping 
        self.data_id_to_annos = defaultdict(list)
        for anno in self.annotations:
            self.data_id_to_annos[ anno["data_id"] ].append(anno)

        '''

        with open(txt,"r") as f:
            data = f.read().split('\n')[:-1]    
        self.image_dir = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/data/MVImgNet_full'  #image_dir 
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

    def get_alpha_mask(self, mask_path):
        # import pdb; pdb.set_trace()
        image = cv2.imread( mask_path) #, cv2.IMREAD_UNCHANGED
        mask = (image[:,:,-1] > 128).astype(np.uint8)
        return mask

    def __getitem__(self, idx):

        out = {}
        # import pdb; pdb.set_trace()
        object_dir = self.data[idx].replace('MVDir', self.image_dir)
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

        # ref_mask_path = os.path.join(mask_dir, ref_mask_name)
        # tar_mask_path = os.path.join(mask_dir, tar_mask_name)
        ref_image_path = os.path.join(object_dir, ref_image_name)
        tar_image_path = os.path.join(object_dir, tar_image_name) 

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path).astype(np.uint8)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        # # add resize
        # ref_image = cv2.resize(ref_image, (512,512))

        tar_image = cv2.imread(tar_image_path).astype(np.uint8)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        # ref_mask = self.get_alpha_mask(ref_mask_path)
        # tar_mask = self.get_alpha_mask(tar_mask_path)
        
        import pdb; pdb.set_trace()
        target_index = os.path.join(tar_image_path.split('/')[-4],tar_image_path.split('/')[-3],tar_image_path.split('/')[-2],tar_image_path.split('/')[-1])
        caption = self.caption_index[target_index]
        
        base_name = os.path.basename(tar_image_path)
        embedding_index = os.path.join(target_index.replace(basename,''), (base_name.split('_')[0]+'.npz'))

        bbox = self.seg2bbox(np.stack([tar_mask,tar_mask,tar_mask],-1))  #将seg map补全成mask
        # 创建一个全黑的图片
        layout = np.zeros((tar_image.shape[0], tar_image.shape[1], 3), dtype=np.uint8)

        # # 已知的bbox坐标：(ymin, xmin, ymax, xmax)
        # padded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)  # 请替换为实际的坐标值
        # 在bbox区域内填充为白色（或其他颜色）
        # 注意：颜色设置为(255, 255, 255)是白色，(R, G, B)格式
        layout[bbox[0]:bbox[2], bbox[1]:bbox[3]] = [255, 255, 255]



        
        out["caption"] = caption
        out["image"] = tar_image
        out['data_id'] = target_index


        image = self.caption_index[idx]

        # data_id = self.data_id_list[idx]
        # out['data_id'] = data_id
        
        file_name = self.data[data_id]['file_name']
        image = self.fetch_image(file_name)
        # out["image"] = image
        out["file_name"] = file_name
        
        # out["caption"] = self.data[data_id]["caption"]

        annos = deepcopy(self.data_id_to_annos[data_id])
        import pdb; pdb.set_trace()
        


        for anno in annos:
            anno["text_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_before",str(embedding_index)), map_location='cpu') 

            anno["image_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_before",str(embedding_index)), map_location='cpu') 
            
            anno["text_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_after",str(embedding_index)), map_location='cpu') 
            
            anno["image_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_after",str(embedding_index)), map_location='cpu') 

        out["annos"] = annos

        return out

    def __len__(self):
        return len(self.data)

from collections import defaultdict
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

        self.annotation_embedding_path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/mvimgnet_tmp' #annotation_embedding_path
        # data point to its annotation mapping 
        self.data_id_to_annos = defaultdict(list)
        # for anno in self.annotations:
        #     self.data_id_to_annos[ anno["data_id"] ].append(anno)
        # image_count = 0
        # for folder in self.data:
        #     image_count += len(glob.glob(folder.replace('MVDir/', self.image_dir)))
        # import pdb; pdb.set_trace()
        
        
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

        base_name = os.path.basename(tar_image_path)
        embedding_index = target_index.replace('.jpg','.npz') #os.path.join(target_index.replace('.jpg','.npz'), (base_name.split('_')[0]+'.npz'))

        # import pdb; pdb.set_trace()
        bbox = self.seg2bbox(np.stack([tar_mask,tar_mask,tar_mask],-1))  #将seg map补全成mask
        # 创建一个全黑的图片
        layout = np.zeros((tar_image.shape[0], tar_image.shape[1], 3), dtype=np.uint8)

        # # 已知的bbox坐标：(ymin, xmin, ymax, xmax)
        # padded_crop = (ymin_new, xmin_new, ymax_new, xmax_new)  # 请替换为实际的坐标值
        # 在bbox区域内填充为白色（或其他颜色）
        # 注意：颜色设置为(255, 255, 255)是白色，(R, G, B)格式
        layout[bbox[0]:bbox[2], bbox[1]:bbox[3]] = [255, 255, 255]
        item_with_collage = self.process_pairs_customized_mvimagenet(ref_image, ref_mask, tar_image, tar_mask,layout)


        out = {}
        out["caption"] = caption
        out["image"] = tar_image
        out['data_id'] = target_index
        out["file_name"] = tar_image_name
        
        annos = {"target_index": None}
        for anno in annos:
            anno["text_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_before",str(embedding_index)), map_location='cpu') 

            anno["image_embedding_before"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_before",str(embedding_index)), map_location='cpu') 
            
            anno["text_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"text_features_after",str(embedding_index)), map_location='cpu') 
            
            anno["image_embedding_after"] = torch.load(os.path.join(self.annotation_embedding_path,"image_features_after",str(embedding_index)), map_location='cpu') 

        out["annos"] = annos

        
        item_with_collage['anno_id'] = target_index
        item_with_collage['txt'] = caption
        sampled_time_steps = self.sample_timestep()
        
        item_with_collage['time_steps'] = sampled_time_steps

        return out




class CDDataset(Base):
    "This only supports instance_json, thus only for O365"
    def __init__(self, image_root, instances_json_path, annotation_embedding_path):
        super().__init__(image_root)

        self.image_root = image_root
        self.instances_json_path = instances_json_path
        self.annotation_embedding_path = annotation_embedding_path
        

        # Load all jsons 
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        clean_annotations(instances_data["annotations"])
        self.instances_data = instances_data

        # Misc  
        self.image_ids = [] # main list for selecting images
        self.image_id_to_filename = {} # file names used to read image
        for image_data in self.instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename

        
        # All category names (including things and stuff)
        self.object_idx_to_name = {} 
        for category_data in self.instances_data['categories']:
            self.object_idx_to_name[category_data['id']] = category_data['name']


        # Add object data from instances and stuff 
        self.image_id_to_objects = defaultdict(list)
        for object_anno in self.instances_data['annotations']:
            image_id = object_anno['data_id']
            self.image_id_to_objects[image_id].append(object_anno)
        

    def getitem(self, index):
        import pdb; pdb.set_trace()
        out = {}
        out['is_det'] = True # indicating this is from detecton data format in TSV

        image_id = self.image_ids[index]
        out['data_id'] = image_id
        
        # Image 
        file_name = self.image_id_to_filename[image_id]
        image = self.fetch_image(file_name)
        out["image"] = image
        out["file_name"] = file_name
    
        # No caption, you need to create one using categories name on fly in TSV 


        annos = deepcopy(self.image_id_to_objects[image_id])
        for anno in annos:
            anno['category_name'] = self.object_idx_to_name[ anno['category_id'] ]

            anno['text_embedding_before'] =  torch.load( os.path.join(self.annotation_embedding_path,"text_features_before",str(anno["id"])), map_location='cpu'  )

            anno['image_embedding_before'] = torch.load( os.path.join(self.annotation_embedding_path,"image_features_before",str(anno["id"])), map_location='cpu'  )

            anno['text_embedding_after'] =  torch.load( os.path.join(self.annotation_embedding_path,"text_features_after",str(anno["id"])), map_location='cpu'  )

            anno['image_embedding_after'] = torch.load( os.path.join(self.annotation_embedding_path,"image_features_after",str(anno["id"])), map_location='cpu'  )

        out['annos'] = annos

        return out 


    def __len__(self):
        return len(self.image_ids)
	






def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


from omegaconf import OmegaConf
# from datasets.mvimagenet_new1 import MVImageNetDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import argparse
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_dataset", type=str, default="grounding", help="grounding is for GoldG, CC3M, SBU etc, detection is for O365")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--total_chunk", type=int, default=1)
    parser.add_argument("--image_root", type=str,default = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/mvimgnet_tmp')
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--annotation_embedding_path", type=str, help='offline processed feature embedding from process_grounding.py script',
    default = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/mvimgnet_tmp')
    parser.add_argument("--tsv_path", type=str, default = 'DATA/mvimgnet_tsv/all.tsv')
    args = parser.parse_args()
    assert args.which_dataset in ["grounding", "detection"]

    image_root = args.image_root
    json_path = args.json_path
    annotation_embedding_path = args.annotation_embedding_path
    tsv_path = args.tsv_path

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/anno.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/SBU/tsv/train-{args.chunk_idx:02d}.tsv"

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/anno.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/CC3M/tsv/train-{args.chunk_idx:02d}.tsv"

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/flickr30k_images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/final_flickr_separateGT_train.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/flickr30k/tsv/train-{args.chunk_idx:02d}.tsv"

    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/images/"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/final_mixed_train_vg.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/embedding_clip/" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/GROUNDING/gqa/tsv/train-{args.chunk_idx:02d}.tsv"


    # image_root = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/images.zip"  # path to image zip file or a image folder 
    # json_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/instances_train.json"  # json annotation used by my normal dataset
    # annotation_embedding_path = "/nobackup2/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/embedding_clip" # it must contain 'image_features' and 'text_features'
    # tsv_path = f"/nobackup3/yuheng-data/diffusion_few_shot/DATA/OBJECTS365/tsv/train-{args.chunk_idx:02d}.tsv"

    # Datasets
    DConf = OmegaConf.load('./configs/datasets.yaml')

    dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)
    # dataset5.getitem(1)

    # image_data = [dataset5]

    # # The ratio of each dataset is adjusted by setting the __len__ 
    # dataset = ConcatDataset(image_data)
    dataset = dataset5
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    # os.makedirs(args.folder, exist_ok=True)

    # if args.which_dataset == "grounding":
    #     dataset = GroundingDataset(image_root,json_path, annotation_embedding_path)
    # else:
    #     dataset = CDDataset(image_root,json_path, annotation_embedding_path)
    # import pdb; pdb.set_trace()

    # dataset = GroundingDataset_mvimgnet(**DConf.Train.MVImageNet)
    # dataset = MVImageNetDataset(**DConf.Train.MVImageNet)

    N = len(dataset)
    print(f'{N} items in total')

    chunk_size = math.ceil(N / args.total_chunk)
    indices = list(split_chunks(list(range(N)), chunk_size))[args.chunk_idx]

    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    writer = TSVWriter(tsv_path)
    # import pdb; pdb.set_trace()
    for i in tqdm(indices):
        item = dataset[i]
        item = item_to_encodable(item)
        # import pdb; pdb.set_trace()
        row = [item['data_id'], json.dumps(item)]
        writer.write(row)

    writer.close()
