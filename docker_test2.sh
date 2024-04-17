# BSUB -o ./bjob_logs/test_4.15_mvimgnet_txt_imageconcat_obj_txtgrounding_grad_trainwithfickr_SBU.%J

# BSUB -q gpu-compute
# BSUB -m "a40-2206.engr.wustl.edu" 

# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA40" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python run_inference_mvimgnet.py