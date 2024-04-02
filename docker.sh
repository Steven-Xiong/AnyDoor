# BSUB -o ./bjob_logs/flickr_train_3.31_mvimgnet.%J

# BSUB -q gpu-compute
# BSUB -m "a100s-2307.engr.wustl.edu" 

# BSUB -gpu "num=2:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
sh scripts/train_mvimagenet.sh