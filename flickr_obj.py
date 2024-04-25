import sys  
sys.path.append('..')

# import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.tsv_dataset import TSVDataset    #这里修改
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

from dataset.concat_dataset import ConCatDataset 
from torch.utils.data.distributed import  DistributedSampler


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
# resume_path = 'checkpoints/control_sd21_ini.ckpt' #'path/to/weight'
resume_path = 'checkpoints/epoch=1-step=8687.ckpt'
batch_size = 4  #16
logger_freq = 500   #1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches=1

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')

# dataset12 = TSVDataset(**DConf.Train.Lvis)

# # import pdb; pdb.set_trace()
# # dataset12.getitem(1)
# # dataset12.get_sample(1)
# image_data = [dataset12]
# # video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
# # tryon_data = [dataset8]
# # threed_data = [dataset5]

# # The ratio of each dataset is adjusted by setting the __len__ 
# dataset = ConcatDataset(image_data)

config = OmegaConf.load('configs/flickr_text_image.yaml') 
# config = DConf
# import pdb; pdb.set_trace()
train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
# ConCatDataset(config.train_dataset_names, 'DATA', train=True, repeats=train_dataset_repeats).getitem(1)
dataset_train = ConCatDataset(config.train_dataset_names, 'DATA', train=True, repeats=train_dataset_repeats)
dataset_train.getitem(1)
# import pdb; pdb.set_trace()
config.distributed = False
sampler = DistributedSampler(dataset_train, seed=123) if config.distributed else None 
# dataloader = DataLoader(dataset_train,  batch_size=batch_size, 
#                                             shuffle=(sampler is None),
#                                             num_workers=config.workers, 
#                                             pin_memory=True, 
#                                             sampler=sampler)


dataloader = DataLoader(dataset_train, num_workers=4, batch_size=batch_size, shuffle=False)

base_save_path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/flickr_ref'
for i, batch in enumerate(dataloader):
    
    print(batch.keys())
    # ref = batch['ref'][0]
    # import pdb; pdb.set_trace()
    if i > 10:
        import pdb; pdb.set_trace()
        print(type(batch['ref']))
        print(type(batch['jpg']))
        print(type(batch['layout']))
        print(batch['ref'][0].shape)
        print(batch['jpg'][0].shape)
        print(batch['layout'][0].shape)
        print(batch['boxes'][0].shape)
        print(batch['masks'][0].shape)

    # plt.imsave(save_path, ref.numpy())
    


    