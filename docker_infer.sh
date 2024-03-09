# BSUB -o ./bjob_logs/Lvis_test2.18.%J

# BSUB -q gpu-compute
# BSUB -m "a100-2207.engr.wustl.edu" 

# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA10080GBPCIe" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
sh scripts/inference.sh