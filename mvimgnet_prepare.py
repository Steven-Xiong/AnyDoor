import sys 
sys.path.append("..") 

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.saliency_modular import SaliencyDataset
from datasets.vipseg import VIPSegDataset
from datasets.mvimagenet_new1 import MVImageNetDataset

from datasets.sam import SAMDataset
from datasets.uvo import UVODataset
from datasets.uvo_val import UVOValDataset
from datasets.mose import MoseDataset
from datasets.vitonhd import VitonHDDataset
from datasets.fashiontryon import FashionTryonDataset
from datasets.lvis_new import LvisDataset    #这里修改
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
# from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
# save_memory = False
# disable_verbosity()
# if save_memory:
#     enable_sliced_attention()

# Configs
# resume_path = 'checkpoints/control_sd21_ini.ckpt' #'path/to/weight'
resume_path = 'checkpoints/epoch=1-step=8687.ckpt'
batch_size = 1  #16
logger_freq = 500   #1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches=1


# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')

dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)


dataset5.getitem(1)
# dataset12.get_sample(1)
image_data = [dataset5]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
# tryon_data = [dataset8]
# threed_data = [dataset5]

# The ratio of each dataset is adjusted by setting the __len__ 
dataset = ConcatDataset(image_data)

dataset.get_sample(17)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
# import pdb; pdb.set_trace()
for idx, batch in enumerate(dataloader):
    import pdb; pdb.set_trace()
    print(batch.keys())


    ref = batch['ref'][0]
    gt = batch['jpg'][0]
    layout = batch['layout'][0]
    save_path = batch['img_path'][0].replace('train2017','train_ref_max')
    save_path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/output/mvimgnet_vis/'
    name = str(idx)+'.png'

    plt.imsave(save_path+'ref'+name, ref.numpy())
    plt.imsave(save_path+'gt'+name, gt.numpy())
    plt.imsave(save_path+'layout'+name, layout.numpy())
    # plt.imsave(save_path+'ref'+name, gt.numpy())