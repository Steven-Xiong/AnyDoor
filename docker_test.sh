# BSUB -o ./bjob_logs/test_4.29_txt_imageconcat_obj_txtgrounding_grad_trainwithcoco_testoncoco.%J

# BSUB -q gpu-compute

# BSUB -gpu "num=1:mode=shared:j_exclusive=yes:gmodel=NVIDIAA40" 
# BSUB -J anydoor_test

source ~/.bashrc
conda activate anydoor
cd /project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor
python run_inference_new1.py