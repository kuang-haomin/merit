import os
import random
import SimpleITK as sitk
import numpy as np
import h5py

def load_nii_data(file_path):
    """加载 .nii.gz 文件并转换为 numpy 数组"""
    itk_image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(itk_image)

def save_to_h5(data_dict, output_path):
    """将数据字典保存为 .h5 文件"""
    with h5py.File(output_path, 'w') as f:
        for key in data_dict:
            f.create_dataset(key, data=data_dict[key])

def process_folders(base_dir,output_dir):
    # 获取所有文件夹路径并排序
    folders = sorted(os.listdir(base_dir))
    # folders.sort()  # 排序文件夹
    
    # 选取前 75% 文件夹并随机选择 5 个
    selected_folders = random.sample(folders[:int(len(folders) * 0.75)], 5)

    data_dict = {}
    for folder in selected_folders:
        folder_path = os.path.join(base_dir, folder)

        # 定义 bm 和 seg 文件路径
        bm_file = os.path.join(folder_path, f"{folder}_bm.nii.gz")
        seg_file = os.path.join(folder_path, f"{folder}_seg.nii.gz")

        # 读取 .nii.gz 文件数据
        image_data = load_nii_data(bm_file)
        label_data = load_nii_data(seg_file)

        # 将数据存入字典
        data_dict['image'] = image_data
        data_dict['label'] = label_data

        # 定义输出文件名
        output_h5_path = os.path.join(output_dir, f"{folder}.h5")

        # 保存字典为 .h5 文件
        save_to_h5(data_dict, output_h5_path)
        print(f"Processed and saved data for folder: {folder}")

if __name__ == "__main__":
    base_dir = '/home/senorgroup/khm/datasets/muRegPro'  # 替换为实际路径
    output_dir = os.path.join(base_dir,"test_h5")
    os.makedirs(output_dir, exist_ok=True)
    process_folders(base_dir,output_dir)
