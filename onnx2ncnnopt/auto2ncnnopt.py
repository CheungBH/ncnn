import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='folder path')

    args = parser.parse_args()
    return args


def main(args):
    path = args.path
    folder_name = path.split("/")[-1]
    for name in os.listdir(path):
        for basename in os.listdir(os.path.join(path,name)):
            if basename.endswith("onnx"):
                param_name = folder_name + '/' + name + '/' + os.path.splitext(basename)[0] + '.param'
                bin_name = folder_name + '/' + name + '/'+ os.path.splitext(basename)[0] + '.bin'
                onnx_name = folder_name + '/' + name + '/'+ basename
                param_opt = folder_name + '/' + name + '/' + os.path.splitext(basename)[0] + '_opt' + '.param'
                bin_opt = folder_name + '/' + name + '/' + os.path.splitext(basename)[0] + '_opt' + '.bin'
                commond_1 = './onnx2ncnn {0} {1} {2}'.format(onnx_name,param_name,bin_name)
                os.system(commond_1)
                command_2 = './ncnnoptimize {0} {1} {2} {3} 65536'.format(param_name, bin_name ,param_opt, bin_opt)
                os.system(command_2)
                os.remove(param_name)
                os.remove(bin_name)

if __name__ == '__main__':
    args = parse_args()
    main(args)
