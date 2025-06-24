
import os
from PIL import Image
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import multiprocessing
import shutil
from multiprocessing import Pool

def get_slice(img_data, axis, img_id):
    if axis == 0:
        return img_data[img_id, :, :]
    elif axis == 1:
        return img_data[:, img_id, :]
    elif axis == 2:
        return img_data[:, :, img_id]
    else:
        raise ValueError("Invalid axis value. Axis must be 0, 1, or 2.")


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
    
    return padded_image.astype(np.uint8)

def select_slices(folder,datapath,des_img,des_lab,axis=0):
  select_num = 20
  image = sitk.ReadImage(os.path.join(datapath,folder,f"{folder}_bm.nii.gz"))
  label = sitk.ReadImage(os.path.join(datapath,folder,f"{folder}_seg.nii.gz"))

  img_data = sitk.GetArrayFromImage(image)
  lab_data = sitk.GetArrayFromImage(label)

  if axis == 0:
    non_zero_slice = [i for i in range(lab_data.shape[axis]) if np.any(lab_data[i,:,:] != 0)]
  elif axis == 1:
    non_zero_slice = [i for i in range(lab_data.shape[axis]) if np.any(lab_data[:,i,:] != 0)]
  elif axis == 2:
    non_zero_slice = [i for i in range(lab_data.shape[axis]) if np.any(lab_data[:,:,i] != 0)]
  else:
    raise ValueError("Invalid axis value. Axis must be 0, 1, or 2.")
     
  # zero_slice = [i for i in range(lab_data.shape[0]) if np.all(lab_data[i,:,:] == 0)]
  # zero_slice = list(set(range(lab_data.shape[axis])) - set(non_zero_slice))

  if len(non_zero_slice) <= select_num:
    selected_non_zero = list(non_zero_slice)
    print(folder,"selected slice num is ",len(selected_non_zero))
  else:
    selected_non_zero = np.random.choice(non_zero_slice, size=select_num, replace=False)

  selected_slices = list(selected_non_zero)

  for img_id in selected_slices:
    out_img = resize_and_pad(get_slice(img_data, axis, img_id),1.0,(192,192))
    out_lab = resize_and_pad(get_slice(lab_data, axis, img_id),1.0,(192,192))
    out_lab = out_lab*255
    out_lab = out_lab.astype(np.uint8)
    slice_image = Image.fromarray(out_img)
    slice_label = Image.fromarray(out_lab)

    # 将图像保存为 PNG 文件
    output_filename = f"{folder}_{axis}_{img_id}.png"
    slice_image.save(os.path.join(des_img, output_filename))
    slice_label.save(os.path.join(des_lab, output_filename))

def check_image_size(image_dir, mask_dir, target_size=(256, 256)):
    # 获取文件夹中的所有文件名
    image_files = set(f for f in os.listdir(image_dir) if f.endswith('.png'))
    mask_files = set(f for f in os.listdir(mask_dir) if f.endswith('.png'))
    
    # 找到所有同名的文件
    common_files = image_files.intersection(mask_files)

    for file in common_files:
        image_path = os.path.join(image_dir, file)
        mask_path = os.path.join(mask_dir, file)

        # 检查 image 文件大小
        with Image.open(image_path) as img:
            if img.size != target_size:
                print(f"Image {file} in 'images' folder has size {img.size}, not {target_size}")

        # 检查 mask 文件大小
        with Image.open(mask_path) as msk:
            if msk.size != target_size:
                print(f"Mask {file} in 'masks' folder has size {msk.size}, not {target_size}")

if __name__ == '__main__':
  datapath = "../../datasets/UltrasoundData_crop_resize"

  des_img = "../../merit_data/pmub/images"
  des_lab = "../../merit_data/pmub/masks"

  if not os.path.exists(des_img):
    os.makedirs(des_img,exist_ok=True)
  if not os.path.exists(des_lab):
    os.makedirs(des_lab,exist_ok=True)

  folders = sorted(os.listdir(datapath))
  trainsize = int(0.75*len(folders))
  print("train is ",trainsize)

  # select_slices(folders[0],datapath,des_img,des_lab,0)

  with multiprocessing.get_context("spawn").Pool(8) as p:
    p.starmap(select_slices, zip(folders[:trainsize],[datapath]*trainsize,[des_img]*trainsize,[des_lab]*trainsize,[0]*trainsize))

  # 设置路径
  image_dir = des_img
  mask_dir = des_lab

  # 调用函数
  check_image_size(image_dir, mask_dir,(192,192))
  # check_image_size(image_dir, mask_dir)

  # for folder in folders:
  #   image = sitk.ReadImage(os.path.join(datapath,folder,f"{folder}_bm.nii.gz"))
  #   label = sitk.ReadImage(os.path.join(datapath,folder,f"{folder}_seg.nii.gz"))

  #   img_data = sitk.GetArrayFromImage(image)
  #   lab_data = sitk.GetArrayFromImage(label)
  #   print(lab_data.shape)

  #   # non_zero_slices = [i for i in range(lab_data.shape[1]) if np.any(lab_data[:,i,:] != 0)]
    

  #   slices = range(lab_data.shape[0])
  #   if len(slices) < 50:
  #     selected_slices = list(slices)
  #   else:
  #     selected_slices = np.linspace(0, len(slices) - 1, 50, dtype=int)

  #   for img_id in selected_slices:
  #     out_img = resize_and_pad(img_data[img_id,:,:],1.5)
  #     out_lab = resize_and_pad(lab_data[img_id,:,:],1.5)
  #     out_lab = out_lab*255
  #     out_lab = out_lab.astype(np.uint8)
  #     slice_image = Image.fromarray(out_img)
  #     slice_label = Image.fromarray(out_lab)

  #     # 将图像保存为 PNG 文件
  #     output_filename = f"{folder}_0_{img_id}.png"
  #     slice_image.save(os.path.join(des_img, output_filename))
  #     slice_label.save(os.path.join(des_lab, output_filename))
    




