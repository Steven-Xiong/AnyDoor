# BSUB -o ./bjob_logs/train_5.20_tsv_concatImageTxt_textdinogrounding_txt_dino_grounding_COCO_addparam.%J

# BSUB -q gpu-compute

# BSUB -gpu "num=2:mode=shared:j_exclusive=yes:gmodel=NVIDIAA10080GBPCIe" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python train_all.py