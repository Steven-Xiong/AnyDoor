file_path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/GROUNDING/flickr30k/tsv/flickr_merged.tsv'

with open(file_path, 'r') as file:
    for _ in range(2):  # 读取前5行
        print(file.readline())