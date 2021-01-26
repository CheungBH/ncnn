import os
import sys


def main(args):
    src_model = args[1]
    if "CNN" in src_model:
        root = "CNN_model"
        if "resnet18" in src_model:
            dest_model_kw = "resnet18"
        elif "mobile" in src_model:
            dest_model_kw = "mobile"
        elif "shuffle" in src_model:
            dest_model_kw = "shuffle"

    elif "det" in src_model:
        root = "det_model"
        dest_model_kw = "yolo"

    elif "CNN" in src_model:
        root = "pose_model"
        if "mobile" in src_model:
            dest_model_kw = "mobilepose"
        elif "alpha" in src_model:
            dest_model_kw = "alphapose"

    elif "hand" in src_model:
        root = "hand_model"
        dest_model_kw = "yolo"

    dest_param = os.path.join(root, dest_model_kw + ".param")
    dest_bin = os.path.join(root, dest_model_kw + ".bin")

    src_param = src_model + ".param"
    src_bin = src_model + ".bin"

    os.system("cp {} {}".format(src_param, dest_param))
    print("cp {} {}".format(src_param, dest_param))
    os.system("cp {} {}".format(src_bin, dest_bin))
    print("cp {} {}".format(src_bin, dest_bin))


if __name__ == '__main__':
    main(sys.argv)


