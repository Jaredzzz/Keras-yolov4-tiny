import os
import numpy as np
import random
import shutil

val_percent = 0.15
train_percent = 0.7
test_percent = 0.15
intensity_image_path = "/data/zjj_workspace/RDNet_data2/intensity_images"
depth_image_path = "/data/zjj_workspace/RDNet_data2/depth_images"
xml_path = "/data/zjj_workspace/RDNet_data2/annotations"

image_savepath = "/data/zjj_workspace/RDNet_data2/dataset/images"
annotation_savepath = "/data/zjj_workspace/RDNet_data2/dataset/annotations"
for type in ["train", "test", "valid"]:
    if os.path.exists(os.path.join(image_savepath, type)):
        for root, dirs, files in os.walk(os.path.join(image_savepath, type), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.mkdir(os.path.join(image_savepath, type))

    if os.path.exists(os.path.join(annotation_savepath, type)):
        for root, dirs, files in os.walk(os.path.join(annotation_savepath, type), topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.mkdir(os.path.join(annotation_savepath, type))

total_image = os.listdir(intensity_image_path)

num = len(total_image)
print("all_num: %s" % num)
# list = range(num)
random.shuffle(total_image)
valid = int(num * val_percent)  # val
test = int(num * test_percent)  # test
count = 0

for image in total_image:
    print(image)
    intensity_image_file_path = os.path.join(intensity_image_path, image)
    depth_image_file_path = os.path.join(depth_image_path, image.replace('Intensity', 'Range'))
    xml_file_path = os.path.join(xml_path, image.replace('tiff', 'xml'))
    if count < valid:
        shutil.copy(intensity_image_file_path, os.path.join(image_savepath, "valid"))
        shutil.copy(depth_image_file_path, os.path.join(image_savepath, "valid"))
        shutil.copy(xml_file_path, os.path.join(annotation_savepath, "valid"))
    elif valid <= count < valid+test:
        shutil.copy(intensity_image_file_path, os.path.join(image_savepath, "test"))
        shutil.copy(depth_image_file_path, os.path.join(image_savepath, "test"))
        shutil.copy(xml_file_path, os.path.join(annotation_savepath, "test"))
    else:
        shutil.copy(intensity_image_file_path, os.path.join(image_savepath, "train"))
        shutil.copy(depth_image_file_path, os.path.join(image_savepath, "train"))
        shutil.copy(xml_file_path, os.path.join(annotation_savepath, "train"))
    count += 1
