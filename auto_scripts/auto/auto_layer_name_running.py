import os


visualize_pipeline = {
    # "CNN_model/mobilenet/2020/CNN-CatDog_mobile_sim-opt-fp16": ["images/catdog"],
    # "CNN_model/mobilenet/2020/mobilenet": ["images/catdog"],
    # "CNN_model/resnet18/2020/CNN-CatDog_resnet18_sim-opt-fp16": ["images/catdog"],
    "model_pose/mobilepose/yoga/yoga_0110/mobilepose.bin": ["images/person", "images/yoga"],
    "model_pose/mobilepose/22/py_mob.param": ["images/person", "images/yoga"],
    "model_pose/mobilepose/22/py_mob_opt.bin": ["images/person", "images/yoga"],
    "model_pose/seresnet101/person/coco_0110/seresnet101_pose.bin": ["images/yoga"]
}


def get_name(file_name):
    with open(file_name, 'r') as f:
        res = f.readlines()
    return res[2].split(' '*10)[2].split(" ")[-1][:-1], res[-1].replace('   ','').split(' ')[5]


def select_exe(name):
    task = name.split("/")[0]
    if "model_CNN" in task:
        exe_name = "CNN"
    elif "model_yolo" in task:
        exe_name = "yolo_detection"
    elif "model_nano" in task:
        exe_name = "nano_detection"
    elif "model_pose" in task:
        exe_name = "pytorch_pose"
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
    os.system(copy_cmd)
    print(copy_cmd)
    inp_name, out_name = get_name(model + ".param")

    for image_src in image_srcs:
        image_dest = generate_result_path(image_src)
        result_dest = "{}/{}".format(image_dest, model.replace("/", "-"))
        os.makedirs(result_dest, exist_ok=True)
        exe_cmd = "./{} {} {}/ {} {}".format(exe_file, image_src, result_dest, inp_name, out_name)
        print(exe_cmd)
        os.system(exe_cmd)

    print("---------------Finish processing model {}-------------------\n".format(idx+1))





