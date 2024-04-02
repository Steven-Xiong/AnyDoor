import pandas as pd
import glob

# # 指定包含TSV文件的目录路径
# path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/GROUNDING/flickr30k/tsv'
# all_files = glob.glob(path + "/*.tsv")

# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0, sep='\t')
#     li.append(df)

# frame = pd.concat(li, axis=0, ignore_index=True)

# # 如果需要，导出到一个新的TSV文件
# frame.to_csv('merged_file.tsv', index=False, sep='\t')

'''
import pandas as pd
import glob

# 指定包含TSV文件的目录路径
path = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/GROUNDING/flickr30k/tsv'
all_files = glob.glob(path + "/*.tsv")

li = []

for i, filename in enumerate(all_files):
    # 读取TSV文件，跳过除了第一个文件以外的文件的表头（header）
    df = pd.read_csv(filename, index_col=None, header=0 if i == 0 else None, sep='\t')
    li.append(df)

# 忽略第一个文件以外的文件的索引，重新设置索引
frame = pd.concat(li, axis=0, ignore_index=True)

# 如果需要，导出到一个新的TSV文件
frame.to_csv('flickr_merged.tsv', index=False, sep='\t')
'''

'''
import pandas as pd
import glob

# 替换为你的TSV文件路径
file_paths = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/GROUNDING/flickr30k/tsv/*.tsv'

# 使用glob模块找到所有的tsv文件
file_list = glob.glob(file_paths)

# 读取第一个文件，创建DataFrame
all_data = pd.read_csv(file_list[0], sep='\t')

# 从第二个文件开始迭代，合并到DataFrame中
for file_path in file_list[1:]:
    data = pd.read_csv(file_path, sep='\t')
    all_data = pd.concat([all_data, data], ignore_index=True)

# 去除重复行
all_data = all_data.drop_duplicates()

# 保存到一个新的TSV文件
all_data.to_csv('merged_data.tsv', sep='\t', index=False)
'''

# 替换为你的TSV文件路径
file_paths = '/project/osprey/scratch/x.zhexiao/edit/LayoutBooth/AnyDoor/DATA/GROUNDING/flickr30k/tsv_small/*.tsv'

# 使用glob模块找到所有的tsv文件
file_list = glob.glob(file_paths)

# 初始化一个空的DataFrame
all_data = pd.DataFrame()

# 遍历所有文件
for i, file_path in enumerate(file_list):
    # 读取文件
    # import pdb; pdb.set_trace()
    data = pd.read_csv(file_path, sep='\t', header=0 if i == 0 else None)
    # 如果不是第一个文件，跳过表头行
    if i > 0:
        data = data[1:]
    # 合并数据
    all_data = pd.concat([all_data, data], ignore_index=False)

# 保存到一个新的TSV文件，不包含索引
all_data.to_csv('merged_data_small_new.tsv', sep='\t', index=False)