# BSUB -o ./bjob_logs/test_4.14_mvimgnet_txt_imageconcat_obj_txtgrounding_grad_trainwithSBU.%J

# BSUB -q gpu-compute
# BSUB -m "a100-2207.engr.wustl.edu" 

# BSUB -gpu "num=1:mode=shared:gmodel=NVIDIAA10080GBPCIe" 
# BSUB -J anydoor

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python run_inference_mvimgnet.py