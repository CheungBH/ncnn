import os

visualize_pipeline = {
    # "CNN_model/mobilenet/2020/CNN-CatDog_mobile_sim-opt-fp16": ["images/catdog"],
    # "CNN_model/mobilenet/2020/mobilenet": ["images/catdog"],
    # "CNN_model/resnet18/2020/CNN-CatDog_resnet18_sim-opt-fp16": ["images/catdog"],
    "pose_model/mobilepose/yoga/yoga_0110/mobilepose.bin": ["images/person", "images/yoga"],
    "pose_model/seresnet101/person/coco_0110/seresnet101_pose.bin": ["images/person", "images/yoga"]
}


def select_exe(name):
    task, structure = name.split("/")[0], name.split("/")[1]
    if "CNN" in task:
        if "mobile" in structure:
            exe_name = "CNN_mobilenet"
        elif "shuffle" in structure:
            exe_name =  "CNN_shuffle"
        elif "resnet18" in structure:
            exe_name =  "CNN_resnet18"
        else:
            raise ValueError("Wrong name of the model!")
    elif "det" in task:
        if "yolo" in structure:
            exe_name = "yolo_detection"
        else:
            raise ValueError("Wrong name of the model!")
    elif "pose_model" in task:
        if "mobile" in structure:
            exe_name = "mobilepose"
        elif "seresnet101" in structure:
            exe_name = "seresnet101_pose"
        else:
            raise ValueError("Wrong name of the model!")
    elif "mm_model_pose" in task:
        if "resnet18" in structure:
            exe_name = "mm_resnet18_pose"
        else:
            raise ValueError("Wrong name of the model!")
    else:
        raise ValueError("Wrong name of the model!")

    return exe_name


def unify_model_name(model_name):
    return model_name.split(".")[0]


def generate_result_path(src_path):
    path_items = src_path.split("/")
    path_items[0] = "vis_" + path_items[0]
    return "/".join(path_items)


for idx, (model, image_srcs) in enumerate(visualize_pipeline.items()):
    model = unify_model_name(model)
    exe_file = select_exe(model)
    copy_cmd = "python copy_model.py {}".format(model)
    # os.system(copy_cmd)

    print(copy_cmd)

    for image_src in image_srcs:
        image_dest = generate_result_path(image_src)
        result_dest = "{}/{}".format(image_dest, model.replace("/", "-"))
        os.makedirs(result_dest, exist_ok=True)
        exe_cmd = "./{} {} {}/".format(exe_file, image_src, result_dest)
        print(exe_cmd)
        os.system(exe_cmd)

    print("---------------Finish processing model {}-------------------\n".format(idx+1))





