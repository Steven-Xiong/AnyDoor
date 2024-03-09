import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset, BaseDataset_t2i
from pycocotools import mask as mask_utils
from lvis import LVIS
from pycocotools.coco import COCO
# import random
#2.5 version, only add text

# class LvisDataset(BaseDataset_t2i):
#     def __init__(self, image_dir, json_path):
#         self.image_dir = image_dir
#         self.json_path = json_path
#         lvis_api = LVIS(json_path)
#         img_ids = sorted(lvis_api.imgs.keys())
#         imgs = lvis_api.load_imgs(img_ids)
#         anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
#         self.data = imgs
#         self.annos = anns
#         self.lvis_api = lvis_api
#         self.size = (512,512)
#         self.clip_size = (224,224)
#         self.dynamic = 0
#         self.coco_json_path = 'data/coco/annotations/captions_train2017.json'
#         self.layout_path = 'data/coco/images/coco_bbox_train'
#         self.box_json_path = 'data/coco/bbox_train.json'
#         self.captions = self.load_captions(self.coco_json_path)
        
#     def register_subset(self, path):
#         data = os.listdir(path)
#         data = [ os.path.join(path, i) for i in data if '.json' in i]
#         self.data = self.data + data
#     # 加load coco caption
#     def load_captions(self,captions_file):

#         with open(captions_file, 'r') as file:
#             data = json.load(file)
#         captions_dict = {}
#         for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
#             image_id = item['image_id']
#             caption = item['caption']
#             if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
#                 captions_dict[image_id] = caption
#         return captions_dict
        
#     def get_sample(self, idx):
#         # ==== get pairs =====
#         image_name = self.data[idx]['coco_url'].split('/')[-1]
#         image_path = os.path.join(self.image_dir, image_name)
#         image = cv2.imread(image_path)
#         ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # import pdb; pdb.set_trace()
#         anno = self.annos[idx]
#         obj_ids = []
#         for i in range(len(anno)):
#             obj = anno[i]
#             area = obj['area']
#             if area > 3600:
#                 obj_ids.append(i)
#         assert len(anno) > 0
#         obj_id = np.random.choice(obj_ids)
#         anno = anno[obj_id]
#         ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
#         # import pdb; pdb.set_trace()
#         tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
        
#         # item_with_collage['txt'] = 'A high resolution, detailed image'     #self.annos[idx]
#         # import pdb; pdb.set_trace()
#         layout_path = os.path.join(self.layout_path,image_name)
#         layout = cv2.imread(layout_path)
#         layout = cv2.cvtColor(layout, cv2.COLOR_BGR2RGB)
#         # print(item_with_collage['layout'].shape)
        
#         item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, layout)  # add layout
#         sampled_time_steps = self.sample_timestep()
#         item_with_collage['time_steps'] = sampled_time_steps
#         item_with_collage['img_path'] = image_path
        
        
#         '''
#         # caption = self.data[idx]['annotations']
#         coco = COCO(self.coco_json_path)
#         # img = coco.loadImgs(idx)[0]
#         annIds = coco.getAnnIds(imgIds=idx)
#         anns = coco.loadAnns(annIds)[0]['caption']
#         # print('idx',idx)
#         # print('annos', anns)
#         '''
        
    
#         # import pdb; pdb.set_trace()
#         # captions = load_captions(self.coco_json_path)
#         annos = self.captions[int(image_name.replace('0','').replace('.jpg',''))]
#         item_with_collage['txt'] = annos
        
#         return item_with_collage

#     def __len__(self):
#         return 20000

#     def check_region_size(self, image, yyxx, ratio, mode = 'max'):
#         pass_flag = True
#         H,W = image.shape[0], image.shape[1]
#         H,W = H * ratio, W * ratio
#         y1,y2,x1,x2 = yyxx
#         h,w = y2-y1,x2-x1
#         if mode == 'max':
#             if h > H or w > W:
#                 pass_flag = False
#         elif mode == 'min':
#             if h < H or w < W:
#                 pass_flag = False
#         return pass_flag

# Use this to do zero123++ augmentation
# class LvisDataset(BaseDataset_t2i):
#     def __init__(self, image_dir, json_path):
#         self.image_dir = image_dir
#         self.json_path = json_path
#         lvis_api = LVIS(json_path)
#         img_ids = sorted(lvis_api.imgs.keys())
#         imgs = lvis_api.load_imgs(img_ids)
#         anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
#         self.data = imgs
#         self.annos = anns
#         self.lvis_api = lvis_api
#         self.size = (512,512)
#         self.clip_size = (224,224)
#         self.dynamic = 0
#         self.coco_json_path = 'data/coco/annotations/captions_train2017.json'
#         self.layout_path = 'data/coco/images/coco_bbox_train'
#         self.box_json_path = 'data/coco/bbox_train.json'
#         self.captions = self.load_captions(self.coco_json_path)
        
#     def register_subset(self, path):
#         data = os.listdir(path)
#         data = [ os.path.join(path, i) for i in data if '.json' in i]
#         self.data = self.data + data
#     # 加load coco caption
#     def load_captions(self,captions_file):

#         with open(captions_file, 'r') as file:
#             data = json.load(file)
#         captions_dict = {}
#         for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
#             image_id = item['image_id']
#             caption = item['caption']
#             if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
#                 captions_dict[image_id] = caption
#         return captions_dict
        
#     def get_sample(self, idx):
#         # ==== get pairs =====
#         # import pdb; pdb.set_trace()
#         image_name = self.data[idx]['coco_url'].split('/')[-1]
#         image_path = os.path.join(self.image_dir, image_name)
#         image = cv2.imread(image_path)
#         ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # import pdb; pdb.set_trace()
#         anno = self.annos[idx]
#         obj_ids = []
#         area_tmp = 0
#         for i in range(len(anno)):
#             obj = anno[i]
#             area = obj['area']
#             if area > 3600 and area > area_tmp:
#                 obj_ids.append(i)
#         assert len(anno) > 0
#         # obj_id = np.random.choice(obj_ids)
#         obj_id = obj_ids[-1]
#         anno = anno[obj_id]
#         ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
#         # import pdb; pdb.set_trace()
#         tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
        
#         # item_with_collage['txt'] = 'A high resolution, detailed image'     #self.annos[idx]
#         # import pdb; pdb.set_trace()
#         layout_path = os.path.join(self.layout_path,image_name)
#         layout = cv2.imread(layout_path)
#         layout = cv2.cvtColor(layout, cv2.COLOR_BGR2RGB)
#         # print(item_with_collage['layout'].shape)
        
#         item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, layout)  # add layout
#         sampled_time_steps = self.sample_timestep()
#         item_with_collage['time_steps'] = sampled_time_steps
#         item_with_collage['img_path'] = image_path
        
        
#         '''
#         # caption = self.data[idx]['annotations']
#         coco = COCO(self.coco_json_path)
#         # img = coco.loadImgs(idx)[0]
#         annIds = coco.getAnnIds(imgIds=idx)
#         anns = coco.loadAnns(annIds)[0]['caption']
#         # print('idx',idx)
#         # print('annos', anns)
#         '''
        
    
#         # import pdb; pdb.set_trace()
#         # captions = load_captions(self.coco_json_path)
#         annos = self.captions[int(image_name.replace('0','').replace('.jpg',''))]
#         item_with_collage['txt'] = annos
        
#         return item_with_collage

#     def __len__(self):
#         return 20000

#     def check_region_size(self, image, yyxx, ratio, mode = 'max'):
#         pass_flag = True
#         H,W = image.shape[0], image.shape[1]
#         H,W = H * ratio, W * ratio
#         y1,y2,x1,x2 = yyxx
#         h,w = y2-y1,x2-x1
#         if mode == 'max':
#             if h > H or w > W:
#                 pass_flag = False
#         elif mode == 'min':
#             if h < H or w < W:
#                 pass_flag = False
#         return pass_flag

class LvisDataset(BaseDataset_t2i):
    def __init__(self, image_dir, json_path):
        self.image_dir = image_dir
        self.json_path = json_path
        lvis_api = LVIS(json_path)
        img_ids = sorted(lvis_api.imgs.keys())
        imgs = lvis_api.load_imgs(img_ids)
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
        
        self.data = imgs
        self.annos = anns
        self.lvis_api = lvis_api
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0
        self.coco_json_path = 'data/coco/annotations/captions_train2017.json'
        self.layout_path = 'data/coco/images/coco_bbox_train0'
        self.box_json_path = 'data/coco/bbox_train.json'
        self.captions = self.load_captions(self.coco_json_path)
        self.ref_dir = 'data/lvis_v1/train_transfered_max'
        
    def register_subset(self, path):
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data
    # 加load coco caption
    def load_captions(self,captions_file):

        with open(captions_file, 'r') as file:
            data = json.load(file)
        captions_dict = {}
        for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
            image_id = item['image_id']
            caption = item['caption']
            if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
                captions_dict[image_id] = caption
        return captions_dict
        
    def get_sample(self, idx):
        # ==== get pairs =====
        # import pdb; pdb.set_trace()
        image_name = self.data[idx]['coco_url'].split('/')[-1]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)

        ref_path = os.path.join(self.ref_dir,image_name)
        ref_image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_image_all = cv2.imread(ref_path)
        # ref_image_all = cv2.imread('/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/data/lvis_v1/val_transfered_max/000000000139.jpg') #ref_path
        ref_image_all = cv2.cvtColor(ref_image_all, cv2.COLOR_BGR2RGB)

        sub_ref_images = []

        # 用两个for循环遍历高度和宽度，步长为320
        H,W = ref_image_all.shape[0], ref_image_all.shape[1] #(960,640)
        for i in range(0, H, int(H/3)):  # 高度方向
            for j in range(0, W, int(W/2)):  # 宽度方向
                # 使用numpy切片操作裁剪子图
                sub_img = ref_image_all[i:i+int(H/3), j:j+int(W/2), :]
                # 将子图添加到列表中
                sub_ref_images.append(sub_img)
        
        random_index = np.random.randint(len(sub_ref_images))
        ref_image = sub_ref_images[random_index]
        
        # ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # import pdb; pdb.set_trace()
        anno = self.annos[idx]
        obj_ids = []
        area_tmp = 0
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']
            if area > 3600 and area > area_tmp:
                obj_ids.append(i)
        assert len(anno) > 0
        # obj_id = np.random.choice(obj_ids)
        obj_id = obj_ids[-1]
        anno = anno[obj_id]
        ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
        # import pdb; pdb.set_trace()
        tar_image, tar_mask = ref_image_original.copy(), ref_mask.copy()
        
        # item_with_collage['txt'] = 'A high resolution, detailed image'     #self.annos[idx]
        # import pdb; pdb.set_trace()
        layout_path = os.path.join(self.layout_path,image_name)
        layout = cv2.imread(layout_path)
        layout = cv2.cvtColor(layout, cv2.COLOR_BGR2RGB)
        # print(item_with_collage['layout'].shape)
        
        # import pdb; pdb.set_trace()
        ref_image = pad_to_square(ref_image, pad_value = 255, random = False)
        ref_image = cv2.resize(ref_image.astype(np.uint8), (224,224) ).astype(np.float32) / 255
        # ref_image = cv2.resize(ref_image.astype(np.uint8), (224,224) ).astype(np.float32) / 127.5 - 1.0
        
        item_with_collage = self.process_pairs_customized(ref_image_original, ref_mask, tar_image, tar_mask, layout)  # add layout    #(500,333,3)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['ref'] = ref_image
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['img_path'] = image_path
        
        
        '''
        # caption = self.data[idx]['annotations']
        coco = COCO(self.coco_json_path)
        # img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=idx)
        anns = coco.loadAnns(annIds)[0]['caption']
        # print('idx',idx)
        # print('annos', anns)
        '''
        
    
        # import pdb; pdb.set_trace()
        # captions = load_captions(self.coco_json_path)
        annos = self.captions[int(image_name.replace('0','').replace('.jpg',''))]
        item_with_collage['txt'] = annos
        
        return item_with_collage

    def __len__(self):
        return len(self.data)

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag


class LvisDataset_val(BaseDataset_t2i):
    def __init__(self, image_dir, json_path):
        self.image_dir = image_dir
        self.json_path = json_path
        lvis_api = LVIS(json_path)
        img_ids = sorted(lvis_api.imgs.keys())
        imgs = lvis_api.load_imgs(img_ids)
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
        
        self.data = imgs
        self.annos = anns
        self.lvis_api = lvis_api
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0
        self.coco_json_path = 'data/coco/annotations/captions_val2017.json'
        self.layout_path = 'data/coco/images/coco_val_bbox'
        self.box_json_path = 'data/coco/bbox1.json'
        self.captions = self.load_captions(self.coco_json_path)
        self.ref_dir = 'data/lvis_v1/val_transfered_max'
        
    def register_subset(self, path):
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data
    # 加load coco caption
    def load_captions(self,captions_file):

        with open(captions_file, 'r') as file:
            data = json.load(file)
        captions_dict = {}
        for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
            image_id = item['image_id']
            caption = item['caption']
            if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
                captions_dict[image_id] = caption
        return captions_dict
        
    def get_sample(self, idx):
        # ==== get pairs =====
        # import pdb; pdb.set_trace()
        image_name = self.data[idx]['coco_url'].split('/')[-1]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)

        ref_path = os.path.join(self.ref_dir,image_name)
        ref_image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_image_all = cv2.imread(ref_path)
        # ref_image_all = cv2.imread('/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/data/lvis_v1/val_transfered_max/000000000139.jpg') #ref_path
        ref_image_all = cv2.cvtColor(ref_image_all, cv2.COLOR_BGR2RGB)

        sub_ref_images = []

        # 用两个for循环遍历高度和宽度，步长为320
        H,W = ref_image_all.shape[0], ref_image_all.shape[1] #(960,640)
        for i in range(0, H, int(H/3)):  # 高度方向
            for j in range(0, W, int(W/2)):  # 宽度方向
                # 使用numpy切片操作裁剪子图
                sub_img = ref_image_all[i:i+int(H/3), j:j+int(W/2), :]
                # 将子图添加到列表中
                sub_ref_images.append(sub_img)
        
        random_index = np.random.randint(len(sub_ref_images))
        ref_image = sub_ref_images[random_index]
        
        # ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # import pdb; pdb.set_trace()
        anno = self.annos[idx]
        obj_ids = []
        area_tmp = 0
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']
            if area > 3600 and area > area_tmp:
                obj_ids.append(i)
        assert len(anno) > 0
        # obj_id = np.random.choice(obj_ids)
        obj_id = obj_ids[-1]
        anno = anno[obj_id]
        ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
        # import pdb; pdb.set_trace()
        tar_image, tar_mask = ref_image_original.copy(), ref_mask.copy()
        
        # item_with_collage['txt'] = 'A high resolution, detailed image'     #self.annos[idx]
        # import pdb; pdb.set_trace()
        layout_path = os.path.join(self.layout_path,image_name)
        layout = cv2.imread(layout_path)
        layout = cv2.cvtColor(layout, cv2.COLOR_BGR2RGB)
        # print(item_with_collage['layout'].shape)
        
        # import pdb; pdb.set_trace()
        ref_image = pad_to_square(ref_image, pad_value = 255, random = False)
        ref_image = cv2.resize(ref_image.astype(np.uint8), (224,224) ).astype(np.float32) / 255
        
        item_with_collage = self.process_pairs_customized(ref_image_original, ref_mask, tar_image, tar_mask, layout)  # add layout    #(500,333,3)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['ref'] = ref_image
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['img_path'] = image_path
        
        
        '''
        # caption = self.data[idx]['annotations']
        coco = COCO(self.coco_json_path)
        # img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=idx)
        anns = coco.loadAnns(annIds)[0]['caption']
        # print('idx',idx)
        # print('annos', anns)
        '''
        
    
        # import pdb; pdb.set_trace()
        # captions = load_captions(self.coco_json_path)
        annos = self.captions[int(image_name.replace('0','').replace('.jpg',''))]
        item_with_collage['txt'] = annos
        
        return item_with_collage

    def __len__(self):
        return len(self.data)

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag
    
# class LvisDataset(BaseDataset_t2i):
#     def __init__(self, image_dir, json_path):
#         self.image_dir = image_dir
#         self.json_path = json_path
#         lvis_api = LVIS(json_path)
#         # import pdb; pdb.set_trace()
#         img_ids = sorted(lvis_api.imgs.keys())
#         imgs = lvis_api.load_imgs(img_ids)
#         anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
#         self.data = imgs
#         self.annos = anns
#         self.lvis_api = lvis_api
#         self.size = (512,512)
#         self.clip_size = (224,224)
#         self.dynamic = 0
#         self.coco_json_path = 'data/coco/annotations/captions_val2017.json'
#         self.layout_path = 'data/coco/images/coco_val_bbox'
#         self.box_json_path = 'data/coco/bbox1.json'
#         self.captions = self.load_captions(self.coco_json_path)
#         self.ref_dir = 'data/lvis_v1/val_transfered_max'
        
#     def register_subset(self, path):
#         data = os.listdir(path)
#         data = [ os.path.join(path, i) for i in data if '.json' in i]
#         self.data = self.data + data
#     # 加load coco caption
#     def load_captions(self,captions_file):

#         with open(captions_file, 'r') as file:
#             data = json.load(file)
#         captions_dict = {}
#         for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
#             image_id = item['image_id']
#             caption = item['caption']
#             if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
#                 captions_dict[image_id] = caption
#         return captions_dict
        
#     def get_sample(self, idx):
#         # ==== get pairs =====
#         # import pdb; pdb.set_trace()
#         image_name = self.data[idx]['coco_url'].split('/')[-1]
#         image_path = os.path.join(self.image_dir, image_name)
#         image = cv2.imread(image_path)
#         ref_path = os.path.join(self.ref_dir,image_name)
#         ref_image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         ref_image_all = cv2.imread(ref_path)
#         # ref_image_all = cv2.imread('/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/data/lvis_v1/val_transfered_max/000000000139.jpg') #ref_path
#         ref_image_all = cv2.cvtColor(ref_image_all, cv2.COLOR_BGR2RGB)

#         sub_ref_images = []

#         # 用两个for循环遍历高度和宽度，步长为320
#         H,W = ref_image_all.shape[0], ref_image_all.shape[1] #(960,640)
#         for i in range(0, H, int(H/3)):  # 高度方向
#             for j in range(0, W, int(W/2)):  # 宽度方向
#                 # 使用numpy切片操作裁剪子图
#                 sub_img = ref_image_all[i:i+320, j:j+320, :]
#                 # 将子图添加到列表中
#                 sub_ref_images.append(sub_img)
        
#         random_index = np.random.randint(len(sub_ref_images))
#         ref_image = sub_ref_images[random_index]

#         # ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # import pdb; pdb.set_trace()
#         anno = self.annos[idx]
#         obj_ids = []
#         area_tmp = 0
#         for i in range(len(anno)):
#             obj = anno[i]
#             area = obj['area']
#             if area > 3600 and area > area_tmp:
#                 obj_ids.append(i)
#         assert len(anno) > 0
#         # obj_id = np.random.choice(obj_ids)
#         obj_id = obj_ids[-1]
#         anno = anno[obj_id]
#         ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
#         # import pdb; pdb.set_trace()
#         tar_image, tar_mask = ref_image_original.copy(), ref_mask.copy()
        
#         # item_with_collage['txt'] = 'A high resolution, detailed image'     #self.annos[idx]
#         # import pdb; pdb.set_trace()
#         layout_path = os.path.join(self.layout_path,image_name)
#         layout = cv2.imread(layout_path)
#         layout = cv2.cvtColor(layout, cv2.COLOR_BGR2RGB)
#         # print(item_with_collage['layout'].shape)
        
#         ref_image = pad_to_square(ref_image, pad_value = 255, random = False)
#         ref_image = cv2.resize(ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) / 255
#         item_with_collage['ref'] = ref_image

#         item_with_collage = self.process_pairs_customized(ref_image, ref_mask, tar_image, tar_mask, layout)  # add layout
#         sampled_time_steps = self.sample_timestep()
        
#         item_with_collage['time_steps'] = sampled_time_steps
#         item_with_collage['img_path'] = image_path
        
        
#         '''
#         # caption = self.data[idx]['annotations']
#         coco = COCO(self.coco_json_path)
#         # img = coco.loadImgs(idx)[0]
#         annIds = coco.getAnnIds(imgIds=idx)
#         anns = coco.loadAnns(annIds)[0]['caption']
#         # print('idx',idx)
#         # print('annos', anns)
#         '''
        
    
#         # import pdb; pdb.set_trace()
#         # captions = load_captions(self.coco_json_path)
#         annos = self.captions[int(image_name.replace('0','').replace('.jpg',''))]
#         item_with_collage['txt'] = annos
        
#         return item_with_collage

#     def __len__(self):
#         return 20000

#     def check_region_size(self, image, yyxx, ratio, mode = 'max'):
#         pass_flag = True
#         H,W = image.shape[0], image.shape[1]
#         H,W = H * ratio, W * ratio
#         y1,y2,x1,x2 = yyxx
#         h,w = y2-y1,x2-x1
#         if mode == 'max':
#             if h > H or w > W:
#                 pass_flag = False
#         elif mode == 'min':
#             if h < H or w < W:
#                 pass_flag = False
#         return pass_flag




class LvisDataset_customized(BaseDataset_t2i):
    def __init__(self, image_dir, json_path):
        self.image_dir = image_dir
        self.json_path = json_path
        lvis_api = LVIS(json_path)
        img_ids = sorted(lvis_api.imgs.keys())
        imgs = lvis_api.load_imgs(img_ids)
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
        self.data = imgs
        self.annos = anns
        self.lvis_api = lvis_api
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0
        self.coco_json_path = 'data/coco/annotations/captions_val2017.json'
        self.layout_path = 'data/coco/images/coco_val_bbox'
        self.box_json_path = 'data/coco/bbox1.json'
        self.captions = self.load_captions(self.coco_json_path)
        self.ref_dir = 'data/lvis_v1/val_transfered_max'

    def register_subset(self, path):
        # import pdb; pdb.set_trace()
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data
    # 加load coco caption
    def load_captions(self,captions_file):

        with open(captions_file, 'r') as file:
            data = json.load(file)
        captions_dict = {}
        for item in data['annotations']:  # 假设 'annotations' 是包含所有字幕信息的键
            image_id = item['image_id']
            caption = item['caption']
            if image_id not in captions_dict:  # 避免覆盖同一个 image_id 的多个标题
                captions_dict[image_id] = caption
        return captions_dict
        
    def get_sample(self, idx):
        # ==== get pairs =====
        
        image_name = self.data[idx]['coco_url'].split('/')[-1]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        import pdb; pdb.set_trace()
        ref_path = os.path.join(self.ref_dir,image_name)
        ref_image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_image_all = cv2.imread('/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/data/lvis_v1/val_transfered_max/000000000139.jpg') #ref_path
        ref_image_all = cv2.cvtColor(ref_image_all, cv2.COLOR_BGR2RGB)

        sub_ref_images = []

        # 用两个for循环遍历高度和宽度，步长为320
        H,W = ref_image_all.shape[0], ref_image_all.shape[1] #(960,640)
        for i in range(0, H, int(H/3)):  # 高度方向
            for j in range(0, W, int(W/2)):  # 宽度方向
                # 使用numpy切片操作裁剪子图
                sub_img = ref_image_all[i:i+320, j:j+320, :]
                # 将子图添加到列表中
                sub_ref_images.append(sub_img)
        
        ref_image = np.random.choice(sub_ref_images)

        # ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()
        anno = self.annos[idx]
        obj_ids = []
        area_tmp = 0
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']
            if area > 3600 and area > area_tmp:
                obj_ids.append(i)
        assert len(anno) > 0
        # obj_id = np.random.choice(obj_ids)
        obj_id = obj_ids[-1]
        anno = anno[obj_id]
        ref_mask = self.lvis_api.ann_to_mask(anno)  #这个是ref condition
        # import pdb; pdb.set_trace()
        tar_image, tar_mask = ref_image_original.copy(), ref_mask.copy()
        
        # item_with_collage['txt'] = 'A high resolution, detailed image'     #self.annos[idx]
        # import pdb; pdb.set_trace()
        layout_path = os.path.join(self.layout_path,image_name)
        layout = cv2.imread(layout_path)
        layout = cv2.cvtColor(layout, cv2.COLOR_BGR2RGB)
        # print(item_with_collage['layout'].shape)
        import pdb; pdb.set_trace()
        #只改变ref_image为直接读取
        ref_image = pad_to_square(ref_image, pad_value = 255, random = False)
        ref_image = cv2.resize(ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) 
        item_with_collage['ref'] = ref_image
        item_with_collage = self.process_pairs_customized(ref_image_original, ref_mask, tar_image, tar_mask, layout)  # add layout
        
        # #进行augmentation
        
        # item_with_collage['ref'] = 
        # item_with_collage['ref'] = 
        # item_with_collage['ref'] = 


        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['img_path'] = image_path
        
        
        
    
        # import pdb; pdb.set_trace()
        # captions = load_captions(self.coco_json_path)
        annos = self.captions[int(image_name.replace('0','').replace('.jpg',''))]
        item_with_collage['txt'] = annos
        
        return item_with_collage

    def __len__(self):
        return 20000

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag
