import glob

# 指定包含.lineidx文件的目录路径
path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/GROUNDING/flickr30k/tsv_small'
all_files = glob.glob(path + "/*.lineidx")

with open('merged_small.lineidx', 'w') as outfile:
    for filename in all_files:
        with open(filename, 'r') as infile:
            for line in infile:
                outfile.write(line)
