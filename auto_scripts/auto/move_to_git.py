import os
import shutil

src_folder = ""
dest_folder = "../../auto_scripts/auto"

file_names = ["auto_convert_darknet.py", "auto_convert_onnx.py", "auto_layer_name_running.py", "copy_model.py",
              "move_ncnn_model.py", "move_to_git.py"]

for file_name in file_names:
    dest_file = os.path.join(dest_folder, file_name)
    shutil.copy(file_name, dest_file)
