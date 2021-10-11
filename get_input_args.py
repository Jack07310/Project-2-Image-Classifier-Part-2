#get_input_args.py
#234567890123456789012345678901234567890123456789012345678901234567890123456789

# Imports here
import argparse

# The command line parser for train.py
def get_input_args_train():

    parser_train = argparse.ArgumentParser()

    parser_train.add_argument('data_directory', nargs = '?',
                              type = str, default='./flowers',
                              help="specify path to the data directory")

    parser_train.add_argument('--save_dir', dest = 'save_dir',
                              type = str, default="./models",
                              help="specify path to the model checkpoint")

    parser_train.add_argument('--arch', dest = 'arch',
                              type = str, default='vgg11',
                              help="provide the model architecture, \
                              e.g. vgg11, vgg13, or vgg16")

    parser_train.add_argument('--learning_rate', dest = 'learning_rate',
                              type = float, default=0.001,
                              help="indicate how fast should the model learn")

    parser_train.add_argument('--hidden_units', dest = 'hidden_units',
                              type = int, default=4096,
                              help="enter number of hidden units \
                              (default is 40966 for vgg11)")

    parser_train.add_argument('--epochs', dest = 'epochs',
                              type = int, default=2,
                              help="enter the number of epochs \
                              (default is 2, avoid more than 5)")

    parser_train.add_argument('--gpu', dest = 'gpu',
                              type = str, default='gpu',
                              help="enter 'gpu' for fast learning \
                                       or 'cpu' otherwise")

    args_train = parser_train.parse_args()

    print("the following arguments or defaults will be used for train.py:")
    print("data_directory:  ", args_train.data_directory)
    print("save_dir:        ", args_train.save_dir)
    print("arch:            ", args_train.arch)
    print("learning_rate:   ", args_train.learning_rate)
    print("hidden_units:    ", args_train.hidden_units)
    print("epochs:          ", args_train.epochs)
    print("gpu:             ", args_train.gpu)

    return(args_train)


# The command line parser for predict.py
def get_input_args_predict():

    parser_predict = argparse.ArgumentParser()

    parser_predict.add_argument('path_to_image', nargs = '?',
                                type = str, 
                                default='./flowers/train/1/image_06734.jpg',
                                help="specify path to the image")
    
    parser_predict.add_argument('path_to_checkpoint', nargs = '?',
                                type = str, default='./models/checkpoint.pth',
                                help="specify path to the model checkpoint")
    
    parser_predict.add_argument('--top_k', dest = 'top_k',
                                type = int, default=1,
                                help="return top K most likely classes \
                                (default 1)")

    parser_predict.add_argument('--category_names', dest = 'category_names',
                                type = str, default='./cat_to_name.json',
                                help="path to mapping of categories to names")

    parser_predict.add_argument('--gpu', dest = 'gpu',
                                type = str, default='gpu',
                                help="enter 'gpu' for fast inference \
                                or 'cpu' otherwise")

    args_predict = parser_predict.parse_args()

    print("the following arguments or defaults will be used for predict.py:")
    print("path_to_image:      ", args_predict.path_to_image)
    print("path_to_checkpoint: ", args_predict.path_to_checkpoint)
    print("top_k:              ", args_predict.top_k)
    print("category_names:     ", args_predict.category_names)
    print("gpu:                ", args_predict.gpu)

    return(args_predict)