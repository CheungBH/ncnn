import os
import shutil

src_folder = "/media/sean/MB155_4/4.9"
dest_folder = "det_model/yolo"

os.makedirs(dest_folder, exist_ok=True)

for folder in os.listdir(src_folder):
    src_sub_folder = os.path.join(src_folder, folder)
    dest_sub_folder = os.path.join(dest_folder, folder)
    os.makedirs(dest_sub_folder)
    for file in os.listdir(src_sub_folder):
        dest_file_name = os.path.join(dest_sub_folder, file)
        src_file_name = os.path.join(src_sub_folder, file)
        if ".bin" in file or ".param" in file:
            shutil.copy(src_file_name, dest_file_name)



