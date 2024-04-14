'''
# import glob
import numpy as np
# txt = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/datasets/Preprocess/mvimagenet_small.txt'
# with open(txt,"r") as f:
#     data = f.read().split('\n')[:-1]

# image_dir = 'data/MVImgNet/'
# image_count = 0
# for folder in data:
#     image_count += len(glob.glob(folder.replace('MVDir/', image_dir)))
# print(image_count) 


array = np.zeros(30,dtype=np.int32)
array[0] = 1
print(array.dntype)
        # item_with_collage['masks'] = array #.reshape(30,1)
        
        '''
import os    
time = '4.13_mvimgnet_trainwithflickr'
dir_path = os.path.join('output',time)
os.makedirs(dir_path,exist_ok=True)
print(dir_path)