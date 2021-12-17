import tensorflow as tf
import os
import unet
import argparse

parser = argparse.ArgumentParser(
    description='Runs the script %(prog)s with the specified parameters',
    usage='%(prog)s model_1 + optional parameters ',
    epilog='Good luck champ!')

parser.add_argument('-ep',
                    help='number of epochs',
                    action="store",
                    type=int)
parser.add_argument('-bs',
                    help='batch size',
                    action="store",
                    type=int)
parser.add_argument('-se',
                    help='steps per epoch',
                    action="store",
                    type=int)
parser.add_argument('-ds',
                    help='names of the data sets to use',
                    action="store",
                    nargs='*')
parser.add_argument('-lr',
                    help='learning rate for the neural network',
                    action="store",
                    type=int)
parser.add_argument('-f',
                    help='factor',
                    action="store",
                    type=int)
parser.add_argument('-p',
                    help='patience',
                    action="store",
                    type=int)
parser.add_argument('-name',
                    help='name of the model',
                    action="store",
                    type=str)
parser.add_argument('-gpu',
                    help='omit if not using gpu',
                    action="store_true",
                    )
parser.add_argument('-pred',
                    help='omit if training',
                    action="store_true",
                    )

def check_gpu(use_gpu):
    if use_gpu:
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def set_params_train(args):
    """ sets parameters and trains the model
    with data defined by user
    """
    print(args)
    if args.gpu:
        check_gpu(True)
    else:
        check_gpu(False)
    checkpoint_file = 'model//' + args.name + '.hdf5' if args.name is not None else \
        'model//unet_testing_dataset.hdf5'
    data_paths = []
    if args.ds is not None:
        for ds in args.ds:
            if '*' in ds:
                name = ds.replace('*', '')
                whole_dir = os.listdir(os.path.join('..', 'destination')) 
                matching = [dir_name.replace('.zip', '') for dir_name in whole_dir if name in dir_name]
                for match in matching:
                    path = os.path.join('..', 'destination', match)
                    data_paths.append(path)
            else:
                path = os.path.join('..', 'destination', ds)
                data_paths.append(path)
    else:
        data_paths.append(os.path.join('..', 'destination'))

    bs = args.bs if args.bs is not None else 1
    ep = args.ep if args.ep is not None else 50
    f = args.f if args.f is not None else .1
    lr = args.lr if args.lr is not None else 0.00001
    p = args.p if args.p is not None else 20
    se = args.se if args.se is not None else 300

    my_unet = unet.myUnet()
    models_path = "stacked_refinement_models"

    try:
        os.mkdir(models_path)
    except Exception as e:
        print(f"Folder '{models_path}' already exists")

    checkpoint_file = os.path.join(models_path, 'stacked_refinement_iteration_0.hdf5')
    my_unet.train(data_paths, checkpoint_file, models_path, batch_size=bs, epochs=ep, factor=f, min_lr=lr, 
                    patience=p, steps_per_epoch=se, no_stacks=5)


def train_net(my_unet):
    data_path = os.path.join('training_images')
    models_path = "stacked_refinement_models"
    checkpoint_file = os.path.join(models_path, 'stacked_refinement_iteration_0.hdf5')

    try:
        os.mkdir(models_path)
    except Exception as e:
        print(f"Folder '{models_path}' already exists")

    my_unet.train(data_path, checkpoint_file, models_path=models_path, epochs=1, no_stacks=5)

def predict_net(my_unet):
    models = []
    for i in range(5):
        models.append(os.path.join('stacked_refinement_models','stacked_refinement_iteration_' + str(i) + '.hdf5'))
    unet.test_predict(my_unet, models)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.pred:
        if args.name is not None:
            predict_net(unet.myUnet())
        else:
            print("Give model name please.")
    else:
        set_params_train(args)
