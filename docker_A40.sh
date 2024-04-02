# BSUB -o ./bjob_logs/train_3.31_A40_try.%J

# BSUB -q gpu-compute
# BSUB -m "a40-2205.engr.wustl.edu" 

# BSUB -gpu "num=4:mode=shared:j_exclusive=yes:gmodel=NVIDIAA40" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
sh scripts/train_mvimagenet.sh