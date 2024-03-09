# BSUB -o ./bjob_logs/Lvis_train_2.18.%J

# BSUB -q gpu-compute
# BSUB -m "a100s-2305.engr.wustl.edu" 

# BSUB -gpu "num=4:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
sh scripts/train_lvis.sh