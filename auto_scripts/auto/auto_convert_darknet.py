import os
src_folder = "/media/sean/MB155_4/4.9"
darknet2ncnn = "/home/sean/Documents/ncnn/build/tools/darknet/darknet2ncnn"
ncnn2opt = "/home/sean/Documents/ncnn/build/tools/ncnnoptimize"

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
        if "cfg" in file:
            cfg = file_path
        elif "weight" in file:
            weight = file_path
        else:
            pass

    print("{} {} {} {} {}".format(darknet2ncnn, cfg, weight, ncnn_param_path, ncnn_bin_path))
    print("{} {} {} {} {} 65536".format(darknet2ncnn, ncnn_param_path, ncnn_bin_path, ncnn_param_opt16_path,
                                            ncnn_bin_opt16_path))

    os.system("{} {} {} {} {}".format(darknet2ncnn, cfg, weight, ncnn_param_path, ncnn_bin_path))
    os.system("{} {} {} {} {} 65536".format(ncnn2opt, ncnn_param_path, ncnn_bin_path, ncnn_param_opt16_path,
                                            ncnn_bin_opt16_path))



