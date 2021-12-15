from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, Callback,  EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import makedirs
import os
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import cv2
import statistics
import argparse
from zipfile import ZipFile
import shutil
from metrics import metrics

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
IMG_MODEL_SIZE = 128


def loader(batch_size, train_path, image_folder, mask_folder, mask_color_mode="grayscale", target_size=(128, 128), save_to_dir=None):
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator(rescale=1. / 255)


    image_generator = image_datagen.flow_from_directory(train_path, classes=[image_folder], class_mode=None,
                                                        color_mode=mask_color_mode, target_size=target_size,
                                                        batch_size=batch_size, save_to_dir=save_to_dir, seed=1)

    mask_generator = mask_datagen.flow_from_directory(train_path, classes=[mask_folder], class_mode=None,
                                                    color_mode=mask_color_mode, target_size=target_size,
                                                    batch_size=batch_size, save_to_dir=save_to_dir, seed=1)
    print("Generating data path:", train_path)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def check_gpu(use_gpu):
    if use_gpu:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class myUnet(Callback):

    def __init__(self, img_rows=IMG_MODEL_SIZE, img_cols=IMG_MODEL_SIZE):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.counter = 0



    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(inputs)

        conv1 = Conv2D(64, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
       

        conv2 = Conv2D(128, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(pool1)

        conv2 = Conv2D(128, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
   

        conv3 = Conv2D(IMG_MODEL_SIZE, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(pool2)

        conv3 = Conv2D(IMG_MODEL_SIZE, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
     

        conv4 = Conv2D(512, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(pool3)

        conv4 = Conv2D(512, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv4)

        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(pool4)

        conv5 = Conv2D(1024, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv5)

        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
   
        # merge6 = merge([drop4, up6], mode = 'concat', concat_axis = 3)
        merge6 = concatenate([drop4, up6], axis=3)
        
        conv6 = Conv2D(512, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge6)

        conv6 = Conv2D(512, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(IMG_MODEL_SIZE, 2, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(IMG_MODEL_SIZE, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge7)

        conv7 = Conv2D(IMG_MODEL_SIZE, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge8)

        conv8 = Conv2D(128, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(merge9)

        conv9 = Conv2D(64, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv9)

        conv9 = Conv2D(2, 3, activation=LeakyReLU(), padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs, conv10)
        # model = Model()
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, checkpoint_file, data_paths, epochs=50, factor=0.1, patience=5, min_lr=0.00001, steps_per_epoch=232, batch_size=1):
        model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=patience, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=factor, patience=patience, min_lr=min_lr, verbose=1)
        print('Fitting model...')

        for path in data_paths:
            # unzip
            with ZipFile(path+'.zip', 'r') as zip:
                name = path.split('\\')[-1]
                if name == path:
                    # for linux paths
                    name = path.split('/')[-1]
                zip.extractall(path.replace(name, ''))
            
            ld = loader(batch_size, path, 'Originals', 'GT')
            model.fit_generator(ld, epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=steps_per_epoch, callbacks=[reduce_lr, early_stopping,model_checkpoint, self])
            
            # remove unzipped to save storage
            try:
                shutil.rmtree(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
                
    def prepare_image_predict(self, input_image):
       img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
       padding = 16
       IMG_MODEL_SIZE = 96
       width = img.shape[1]
       height = img.shape[0]
       delta_x = width // IMG_MODEL_SIZE
       delta_y = height // IMG_MODEL_SIZE
       remx = width % IMG_MODEL_SIZE
       remy = height % IMG_MODEL_SIZE
       parts = []
       
       border_width = width + padding + IMG_MODEL_SIZE - remx + padding
       border_height = height + padding + IMG_MODEL_SIZE - remy + padding
       
       border = np.zeros([border_height, border_width], dtype=np.uint8)
       border.fill(255)  # or img[:] = 255
       border[padding:padding+height,padding:padding+width] = img
       cv2.imwrite(os.path.join('..', 'testimages', 'border.png'), border)
       if remx > 0:
           delta_x = delta_x + 1
       if remy > 0:
           delta_y = delta_y + 1
       for x in range(delta_x):
           xinit = x * IMG_MODEL_SIZE
           for y in range(delta_y):
               yinit = y * IMG_MODEL_SIZE
               part = border[yinit:yinit + 128,xinit:xinit +128]
               parts.append(part.astype('float32'))
               
       #for i,part in enumerate(parts):
           #cv2.imwrite(os.path.join('..', 'testimages', str(i) +'.png'), part)
           
       return np.asarray(parts), (img.shape[0], img.shape[1])



    def restore_image(self, parts, dim):
        padding = 16
        IMG_MODEL_SIZE = 96
        width = dim[1]
        height = dim[0]
        result = np.zeros([height, width, 1], dtype=np.uint8)
        result.fill(255)  # or img[:] = 255
        delta_x = width // IMG_MODEL_SIZE
        delta_y = height // IMG_MODEL_SIZE
        remx = width % IMG_MODEL_SIZE
        remy = height % IMG_MODEL_SIZE
        index = 0
        border_width = width + padding + IMG_MODEL_SIZE - remx + padding
        border_height = height + padding + IMG_MODEL_SIZE - remy + padding
       
        border = np.zeros([border_height, border_width, 1], dtype=np.uint8)
        border.fill(0)  # or img[:] = 255
        if remx > 0:
           delta_x = delta_x + 1
        if remy > 0:
           delta_y = delta_y + 1
        for x in range(delta_x):
            xinit = x * IMG_MODEL_SIZE
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                #cv2.imwrite(os.path.join('..', 'testimages', str(index) +'_read.png'), parts[index]*255)
                border[yinit:yinit + IMG_MODEL_SIZE,xinit:xinit + IMG_MODEL_SIZE] = parts[index][padding:IMG_MODEL_SIZE + padding,padding:IMG_MODEL_SIZE + padding] * 255
                index += 1
            
        result = border[:height,:width]
        return result


    def binarise_image(self, model_weights, input_image):
        print("loading image")
        parts, dim = self.prepare_image_predict(input_image=input_image)
        model = self.get_unet()

        model.load_weights(model_weights)
        print('predicting test data')
        imgs_mask_test = model.predict(parts, batch_size=1, verbose=1)

        return self.restore_image(imgs_mask_test, dim)



def test_predict(u_net, model):
    images = os.listdir(os.path.join('images'))
    results = []
    for image in images:
        ground_truth = cv2.imread(os.path.join('GT', image[:-4] + '.png'), cv2.IMREAD_GRAYSCALE)
        current_image = os.path.join('images', image)
        result_unet = u_net.binarise_image(model_weights=model, input_image=current_image)
        result_otsu = threshold_otsu(result_unet)
        result_unet_otsu = ((result_unet > result_otsu) * 255).astype(np.uint8)
        img_true = np.array(ground_truth).ravel()

        cv2.imwrite(os.path.join('results', image[:-4] + '_' + 'unet_.png'), result_unet)
        cv2.imwrite(os.path.join('results', image[:-4] + '_' + 'unet_otsu_.png'), result_unet_otsu)
        result_unet_otsu = result_unet_otsu[:,:,0]

        fmeasure, pfmeasure, psnr, nrm, mpm, drd = metrics(result_unet_otsu, ground_truth)
        results.append([['F-Measure', fmeasure], ['pf-Measure', pfmeasure], 
        ['PSNR', psnr], ['NRM', nrm], ['MPM', mpm], ['DRD', drd]])

    measures = []
    for i in range(len(results[0])):
        meas_mean = statistics.mean([row[i][1] for row in results])
        measures.append([results[0][i][0], meas_mean])
    headers = ['Measure', 'Score']
    print(pd.DataFrame(measures, None, headers))

def set_params_train(args):
    """ sets parameters and trains the model
    with data defined by user
    """
    if args.gpu:
        check_gpu(True)
    else:
        check_gpu(False)

    data_paths = []
    names = ""
    if args.ds is not None:
        names = '_'.join(args.ds)
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
    
    

    if args.name is not None:
        checkpoint_file = os.path.join('model', args.name + '.hdf5')
    else:
        checkpoint_file = os.path.join('model', f'{ep}_{se}_{lr}_{p}_{f}_{names}.hdf5' )
    
    
    my_unet = myUnet()
    my_unet.train(checkpoint_file, data_paths, batch_size=bs, epochs=ep, factor=f, min_lr=lr, patience=p, steps_per_epoch=se)

def load_model_predict(model_name='unet_testing_dataset.hdf5'):
    """ loads the model specified by input arg
    and predicts from images folder
    """
    my_unet = myUnet()
    model = os.path.join('model', model_name + '.hdf5')
    test_predict(my_unet, model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.pred:
        if args.name is not None:
            load_model_predict(args.name)
        else:
            print("Give a model name please.")
    else:
        set_params_train(args)
 
