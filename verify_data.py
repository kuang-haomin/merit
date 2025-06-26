import os
import random
import SimpleITK as sitk
import numpy as np
import h5py

base_dir = "../merit_data/muregpro/val_h5"
files = os.listdir(base_dir)
for file in files:
  filepath = os.path.join(base_dir,file)
  data = h5py.File(filepath)
  image, label = data['image'][:], data['label'][:]
  image = np.array(image)/255.0
  label = np.array(label)
  for i in range(image.shape[2]):
    if image[:,:,i].max() < 0.1:
      print(file,i,"error")
    if image[:,:,i].max() > 0.8 and label[:,:,i].max()==0:
      print(file,i,"image > 0.8, but label is 0")
    # print("image max is ",image[:,:,i].max())
    # print("label max is ",label[:,:,i].max())