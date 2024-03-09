
import glob

txt = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/datasets/Preprocess/mvimagenet_small.txt'
with open(txt,"r") as f:
    data = f.read().split('\n')[:-1]

image_dir = 'data/MVImgNet/'
image_count = 0
for folder in data:
    image_count += len(glob.glob(folder.replace('MVDir/', image_dir)))
print(image_count) 