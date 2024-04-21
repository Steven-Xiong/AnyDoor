# BSUB -o ./bjob_logs/train_4.21_tsv_concatImageTxt_objstxtembedding_grad_fixed_3data_2306.%J

# BSUB -q gpu-compute
# BSUB -m "a100s-2306.engr.wustl.edu" 

# BSUB -gpu "num=3:mode=shared:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python train_all_3cards.py