# BSUB -o ./bjob_logs/train_4.25_tsv_concatImageTxt_textdinogrounding_grad_fixed_COCO.%J

# BSUB -q gpu-compute
# BSUB -m "a100s-2305.engr.wustl.edu" 

# BSUB -gpu "num=4:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python train_all.py