import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from skimage.transform import resize
import SimpleITK as sitk


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class muregpro_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None, mode = "train"):
        self.transform = transform  # using transform in torch!
        self.split = split
        print("loading data from the directory :",base_dir)
        images = sorted(glob(os.path.join(base_dir, "images/*.png")))
        masks = sorted(glob(os.path.join(base_dir, "masks/*.png")))
        self.sample_list = [os.path.basename(image) for image in images]

        self.name_list = images
        self.label_list = masks
        self.mode = mode
        self.data_dir = base_dir
        self.nclass = nclass
        if split != "train":
            self.sample_list = sorted(os.listdir(base_dir))

    def __len__(self):
        # length = 16
        # return length
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            """Get the images"""
            img_path = self.name_list[idx]
            msk_path = self.label_list[idx]

            img = Image.open(img_path)
            mask = Image.open(msk_path)

            # 将 img 转换为 numpy 数组并按给定每个像素值除以 255
            image = np.array(img)/255.0

            # 将 mask 转换为 numpy 数组，并将 255 的值转换为 1，其他值保持不变
            label = np.array(mask)
            label[label == 255] = 1.0

            image = image.astype(np.float32)
            label = label.astype(np.float32)

        else:
            vol_name = self.sample_list[idx]
            filepath = os.path.join(self.data_dir, vol_name)
            
            if os.path.isdir(filepath):  # 如果 vol_name 是一个文件夹
                # 构造 image 和 label 的文件路径
                image_path = os.path.join(filepath, f"{vol_name}_bm.nii.gz")
                label_path = os.path.join(filepath, f"{vol_name}_seg.nii.gz")
                
                # 使用 SimpleITK 加载 .nii.gz 文件
                image_nii = sitk.ReadImage(image_path)  # 读取 image 文件
                label_nii = sitk.ReadImage(label_path)  # 读取 label 文件
                
                # 将 SimpleITK 图像对象转换为 NumPy 数组
                image = sitk.GetArrayFromImage(image_nii)  # 转换为 NumPy 数组
                label = sitk.GetArrayFromImage(label_nii)  # 转换为 NumPy 数组
            else:
                # 如果不是文件夹，则读取原来的 h5 文件（如原始代码）
                data = h5py.File(filepath)
                image, label = data['image'][:], data['label'][:]
            # data = h5py.File(filepath)
            # image, label = data['image'][:], data['label'][:]
            label = np.array(label)
            # print(label.sum())
            image = rearrange_and_pad(image)
            label = rearrange_and_pad(label)
            # print(label.sum())
            # for i in range(image.shape[0]):
            #     if image[i,:,:].max() < 0.1:
            #         print(vol_name,i,"error, image max is", image[i,:,:].max())

            image = np.array(image)/255.0
            image = image.astype(np.float32)
            label = label.astype(np.float32)    

            #image = np.reshape(image, (image.shape[2], 512, 512))
            #label = np.reshape(label, (label.shape[2], 512, 512))

        if self.nclass == 9:
            label[label==5]= 0
            label[label==9]= 0
            label[label==10]= 0
            label[label==12]= 0
            label[label==13]= 0
            label[label==11]= 5
        # print("image max is ",image.max(), " ,label max is",label.max())
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            #print(image.shape)
            #image = np.reshape(image, (512, 512))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            #label = np.reshape(label, (512, 512))
            
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            #image = np.reshape(image, (image.shape[2], 512, 512))
            #label = np.reshape(label, (label.shape[2], 512, 512))
            #label[label==5]= 0
            #label[label==9]= 0
            #label[label==10]= 0
            #label[label==12]= 0
            #label[label==13]= 0
            #label[label==11]= 5

        if self.nclass == 9:
            label[label==5]= 0
            label[label==9]= 0
            label[label==10]= 0
            label[label==12]= 0
            label[label==13]= 0
            label[label==11]= 5
        # print("image max is ",image.max(), " ,label max is",label.max())
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def rearrange_and_pad(input_array):
    """
    将输入的 3D numpy 数组的第三个轴调整到第一个轴，
    然后对第一个轴的每个切片 pad 成 128x128 的 shape。

    参数:
    input_array (numpy.ndarray): 输入的 3D numpy 数组，形状为 (x, y, z)

    返回:
    numpy.ndarray: 经过处理后的 3D numpy 数组，形状为 (z, 128, 128)
    """
    # 将第三个轴调整到第一个轴
    rearranged_array = np.moveaxis(input_array, 2, 0)  # 从 (x, y, z) 到 (z, x, y)

    # 创建一个新的数组来存储 pad 之后的结果
    padded_array = []

    # 对每一个切片进行 padding
    for i in range(rearranged_array.shape[0]):
        slice_data = rearranged_array[i]
        padded_slice = resize_and_pad(slice_data, resize_factor=1.0, target_size=(128, 128))
        # 将 pad 后的切片添加到列表中
        padded_array.append(padded_slice)
    # 将列表转换回 3D numpy 数组
    padded_array = np.array(padded_array)

    return padded_array

def resize_and_pad(image, resize_factor=1.0, target_size=(256, 256)):
    """
    Resize the input image by a specified factor and then pad it to the target size (256x256).
    
    Parameters:
    - image: 2D NumPy array (input image).
    - resize_factor: The factor by which to resize the image. 1.0 means no resizing.
    - target_size: The final desired size of the image after padding.
    
    Returns:
    - The resized and padded image, with shape (256, 256).
    """
    # 计算目标缩放尺寸
    if resize_factor == 1:
        resized_image = image
    else:
        resized_image = resize(image, 
                           (int(image.shape[0] * resize_factor), int(image.shape[1] * resize_factor)),
                           mode='constant', 
                           preserve_range=True)  # 保证范围不变
    
    # 如果缩放后图像大于目标尺寸，裁剪中心区域
    # if resized_image.shape[0] > target_size[0] or resized_image.shape[1] > target_size[1]:
    if resized_image.shape[0] > target_size[0]:
      start_y = (resized_image.shape[0] - target_size[0]) // 2
      end_y = start_y + target_size[0]
    else:
      start_y = 0
      end_y = resized_image.shape[0]
    if resized_image.shape[1] > target_size[1]:
      start_x = (resized_image.shape[1] - target_size[1]) // 2
      end_x = start_x + target_size[1]
    else:
      start_x = 0
      end_x = resized_image.shape[1]
    cropped_image = resized_image[start_y:end_y, start_x:end_x]
        # return cropped_image.astype(np.uint8)

    # 计算填充的大小
    pad_y = (target_size[0] - cropped_image.shape[0]) // 2
    pad_x = (target_size[1] - cropped_image.shape[1]) // 2
    
    # 如果缩放后图像小于目标尺寸，需要填充
    padded_image = np.pad(cropped_image, 
                          ((pad_y, target_size[0] - cropped_image.shape[0] - pad_y),
                           (pad_x, target_size[1] - cropped_image.shape[1] - pad_x)),
                          mode='constant', constant_values=0)
    
    return padded_image