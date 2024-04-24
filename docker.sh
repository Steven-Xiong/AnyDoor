# BSUB -o ./bjob_logs/train_4.23_tsv_concatImageTxt_objstxtembedding_grad_fixed_coco_noref.%J

# BSUB -q gpu-compute
# BSUB -m "a100s-2307.engr.wustl.edu" 

# BSUB -gpu "num=3:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
sh scripts/train.sh