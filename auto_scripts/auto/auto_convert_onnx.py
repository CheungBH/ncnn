import os
src_folder = "/home/sean/Downloads/nanodet_onnx"
onnx2ncnn = "/home/sean/Desktop/ncnn/build/tools/onnx/onnx2ncnn"
ncnn2opt = "/home/sean//ncnn/build/tools/ncnnoptimize"

# curr_path = os.getcwd()
# src_folder

for folder in os.listdir(src_folder):
    sub_folder = os.path.join(src_folder, folder)
    ncnn_bin_path = os.path.join(sub_folder, "ncnn.bin")
    ncnn_param_path = os.path.join(sub_folder, "ncnn.param")
    # ncnn_bin_opt_path = os.path.join(sub_folder, "ncnn_opt.bin")
    # ncnn_param_opt_path = os.path.join(sub_folder, "ncnn_opt.param")
    ncnn_bin_opt16_path = os.path.join(sub_folder, "ncnn_opt-fp16.bin")
    ncnn_param_opt16_path = os.path.join(sub_folder, "ncnn_opt-fp16.param")

    for file in os.listdir(sub_folder):
        file_path = os.path.join(sub_folder, file)
        if ".onnx" in file:
            onnx = file_path
        else:
            pass

    onnx_cmd = "{} {} {} {}".format(onnx2ncnn, onnx, ncnn_param_path, ncnn_bin_path)
    ncnn_opt_cmd = "{} {} {} {} {} 65536".format(ncnn2opt, ncnn_param_path, ncnn_bin_path, ncnn_param_opt16_path,
                                            ncnn_bin_opt16_path)
    print(onnx_cmd)
    print(ncnn_opt_cmd)
    os.system(onnx_cmd)
    os.system(ncnn_opt_cmd)
