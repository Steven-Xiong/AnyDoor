# BSUB -o ./bjob_logs/test_4.1.%J

# BSUB -q gpu-compute
# BSUB -m "a100-2207.engr.wustl.edu" 

# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA10080GBPCIe" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python run_inference_new.py