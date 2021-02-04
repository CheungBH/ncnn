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
        else:
            raise ValueError("Wrong model path")

    elif "det" in src_model:
        root = "det_model"
        dest_model_kw = "yolo"

    elif "pose" in src_model:
        root = "pose_model"
        if "mobile" in src_model:
            dest_model_kw = "mobilepose"
        elif "seresnet101" in src_model:
            dest_model_kw = "seresenet101"
        else:
            raise ValueError("Wrong model path")

    else:
        raise ValueError("Wrong model path")

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


