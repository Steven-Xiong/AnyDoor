import pytorch_lightning as pl
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
resume_path = 'checkpoints/epoch=1-step=8687.ckpt'   #'checkpoints/epoch=1-step=8687.ckpt'
batch_size = 16
logger_freq = 200   #1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 4
accumulate_grad_batches=1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/anydoor.yaml').cpu()
#这里要舍弃不用的key,val

model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict = False)  #设置strict = False
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
# DConf = OmegaConf.load('./configs/datasets.yaml')

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
# dataset_train.getitem(1)
# import pdb; pdb.set_trace()

# config.distributed = True
# sampler = DistributedSampler(dataset_train, seed=123) if config.distributed else None 

dataloader = DataLoader(dataset_train, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=n_gpus, strategy="ddp", precision=16, accelerator="gpu", callbacks=[logger], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)

# Train!
trainer.fit(model, dataloader)