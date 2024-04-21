# BSUB -o ./bjob_logs/test_4.21_nofref_testonsbu.%J

# BSUB -q gpu-compute
# BSUB -m "a40-2206.engr.wustl.edu" 

# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA40" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python run_inference_new1.py