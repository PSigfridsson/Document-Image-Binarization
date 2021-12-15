from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, Callback,  EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import sigmoid
import numpy as np
from matplotlib import pyplot as plt
from os import makedirs
import pandas as pd
import os
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from sklearn.metrics import jaccard_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import cv2
import statistics
import gt_construction

USE_GPU = True
IMG_MODEL_SIZE = 128


def image_add(img_a, img_b):
    result = np.zeros([img_a.shape[0], img_a.shape[1]], dtype=np.uint8)

    for y, row in enumerate(img_a):
        for x, pixel in enumerate(row):
            if int(img_a[y,x])+int(img_b[y,x]) > 255:
                result[y,x] = 255
            elif int(img_a[y,x])+int(img_b[y,x]) < 0:
                result[y,x] = 0
            else:
                result[y,x] = img_a[y,x]+img_b[y,x]

    return result


def remove_negative_pixels(img):
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            if pixel < 0:
                img[y,x] = 0
    return img



def loader(batch_size, train_path, image_folder, mask_folder, original_image_folder, mask_color_mode="grayscale", target_size=(128, 128), save_to_dir=None):
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    mask_datagen = ImageDataGenerator(rescale=1. / 255)

    image_generator = image_datagen.flow_from_directory(train_path, classes=[image_folder], class_mode=None,
                                                        color_mode=mask_color_mode, target_size=target_size,
                                                        batch_size=batch_size, save_to_dir=save_to_dir, seed=1)

    mask_generator = mask_datagen.flow_from_directory(train_path, classes=[mask_folder], class_mode=None,
                                                      color_mode=mask_color_mode, target_size=target_size,
                                                      batch_size=batch_size, save_to_dir=save_to_dir, seed=1)

    original_image_generator = image_datagen.flow_from_directory(train_path, classes=[original_image_folder], class_mode=None,
                                                        color_mode=mask_color_mode, target_size=target_size,
                                                        batch_size=batch_size, save_to_dir=save_to_dir, seed=1)
    train_generator = zip(image_generator, mask_generator, original_image_generator)
    counter = 0
    for (img, mask, og) in train_generator:
        # allowed neg_e to be negative??
        mask = mask[0,:,:,:]
        mask = ((mask > 0.5)).astype(np.uint8)
        img = img[0,:,:,:]
        og = og[0,:,:,:]

        grayscale_gt = gt_construction.generate_grayscale_gt(og, mask)

        neg_e = grayscale_gt-img
        #neg_e = remove_negative_pixels(neg_e)

        # cv2.imwrite(os.path.join('results', str(counter) + 'img.png'), 255*img)
        # cv2.imwrite(os.path.join('results', str(counter) + 'og.png'), 255*og)
        # cv2.imwrite(os.path.join('results', str(counter) + 'grayscale_gt.png'), 255*grayscale_gt)
        # cv2.imwrite(os.path.join('results', str(counter) + 'neg_e.png'), 255*neg_e)
        
        neg_e = np.expand_dims(neg_e, axis=0)
        img = np.expand_dims(img, axis=0)
        counter += 1
        yield (img, neg_e)

def check_gpu():
    if USE_GPU:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def custom_activation(x):
    return (sigmoid(x) * 2) - 1

class myUnet(Callback):

    def __init__(self, img_rows=IMG_MODEL_SIZE, img_cols=IMG_MODEL_SIZE):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        # Code here what you want each time an epoch ends
        print('--- on_epoch_end ---')
        #self.save_epoch_results()

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(inputs)
        conv1 = BatchNormalization()(conv1)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv1)
        conv1 = BatchNormalization()(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool1)
        conv2 = BatchNormalization()(conv2)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv2)
        conv2 = BatchNormalization()(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool2)
        conv3 = BatchNormalization()(conv3)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv3)
        conv3 = BatchNormalization()(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(drop5))
        print('up6 ' + str(up6))
        print('drop4 ' + str(drop4))
        # merge6 = merge([drop4, up6], mode = 'concat', concat_axis = 3)
        merge6 = concatenate([drop4, up6], axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(IMG_MODEL_SIZE, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv6))
        # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        merge7 = concatenate([conv3, up7], axis=3)

        conv7 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(IMG_MODEL_SIZE, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv7))
        # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        merge8 = concatenate([conv2, up8], axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='glorot_uniform')(
            UpSampling2D(size=(2, 2))(conv8))
        # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        merge9 = concatenate([conv1, up9], axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv10 = Conv2D(1, 1, activation=custom_activation)(conv9)

        model = Model(inputs, conv10)
        # model = Model()

        model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error', metrics=['accuracy']) # mean_absolute_error binary_crossentropy

        return model


    def train(self, data_path, checkpoint_file, models_path, epochs=50, no_stacks=1):
        model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=20, verbose=1, monitor='loss')
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1, monitor='loss')

        print('Fitting model...')
        if no_stacks == 1:
            ld = loader(1, data_path, 'Originals', 'GT', 'Originals')
            model.fit_generator(ld, epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=218, callbacks=[reduce_lr, early_stopping, model_checkpoint, self])
        else:
            self.stacked_refinement(model, data_path, epochs, no_stacks, reduce_lr, early_stopping, model_checkpoint, models_path)
            #self.recursive_refinement(model, data_path, epochs, no_stacks, reduce_lr, early_stopping, model_checkpoint)

    def recursive_refinement(self, model, data_path, epochs, no_recursions, reduce_lr, early_stopping, model_checkpoint):
        pred_originals = 'Originals'

        for recursion in range(no_recursions):
            ld = loader(1, data_path, pred_originals, 'GT', 'Originals')
            model.fit_generator(ld, epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=218, callbacks=[reduce_lr, early_stopping, model_checkpoint, self])

            new_originals_folder = os.path.join(data_path, f'Originals_{recursion}')

            try:
                os.mkdir(new_originals_folder)
            except Exception as e:
                print(f"Folder '{new_originals_folder}' already exists!")

                for file in os.listdir(new_originals_folder):
                    os.remove(os.path.join(new_originals_folder, file))

            images_path = os.path.join(data_path, pred_originals)
            for image in os.listdir(images_path):
                image_path = os.path.join(images_path, image)
                xu = self.binarise_image(input_image=image_path, name=image[:-4], model=model)

                new_image_path = os.path.join(new_originals_folder, image)
                cv2.imwrite(new_image_path, xu)

            early_stopping = EarlyStopping(patience=20, verbose=1, monitor='loss')
            reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1, monitor='loss')
            pred_originals = f'Originals_{recursion}'

    def stacked_refinement(self, model, data_path, epochs, no_stacks, reduce_lr, early_stopping, model_checkpoint, models_path):
        pred_originals = 'Originals'

        for stack in range(no_stacks):
            ld = loader(1, data_path, pred_originals, 'GT', 'Originals')
            model.fit_generator(ld, epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=10000, callbacks=[reduce_lr, early_stopping, model_checkpoint, self])

            new_originals_folder = os.path.join(data_path, f'Originals_{stack}')

            try:
                os.mkdir(new_originals_folder)
            except Exception as e:
                print(f"Folder '{new_originals_folder}' already exists!")

                for file in os.listdir(new_originals_folder):
                    os.remove(os.path.join(new_originals_folder, file))

            images_path = os.path.join(data_path, pred_originals)
            for image in os.listdir(images_path):
                image_path = os.path.join(images_path, image)
                xu = self.binarise_image(input_image=image_path, name=image[:-4], model=model)

                new_image_path = os.path.join(new_originals_folder, image)
                cv2.imwrite(new_image_path, xu)

            model = self.get_unet()
            checkpoint_file = os.path.join(models_path, f'stacked_refinement_iteration_{stack+1}.hdf5')
            model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True)
            early_stopping = EarlyStopping(patience=20, verbose=1, monitor='loss')
            reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1, monitor='loss')
            pred_originals = f'Originals_{stack}'

    def prepare_image_predict_original(self, input_image):
        img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        width = img.shape[1]
        height = img.shape[0]
        delta_x = width // IMG_MODEL_SIZE
        delta_y = height // IMG_MODEL_SIZE
        remx = width % IMG_MODEL_SIZE
        remy = height % IMG_MODEL_SIZE
        print('remy', remy)
        parts = []
        border = np.zeros([IMG_MODEL_SIZE, IMG_MODEL_SIZE], dtype=np.uint8)
        border.fill(255)  # or img[:] = 255

        for x in range(delta_x):
            xinit = x * IMG_MODEL_SIZE
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                part = img[yinit:yinit + IMG_MODEL_SIZE, xinit:xinit+IMG_MODEL_SIZE]
                parts.append(part.astype('float32') / 255)
            if remy > 0:
                border.fill(255)
                border[:remy, :] = img[height-remy:height, xinit:xinit+IMG_MODEL_SIZE]
                parts.append(border.astype('float32') / 255)

        if remx > 0:
            xinit = width - remx
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                border.fill(255)
                border[:, :remx] = img[yinit:yinit + IMG_MODEL_SIZE, xinit:width]
                parts.append(border.astype('float32') / 255)
            if remy > 0:
                border.fill(255)
                border[:remy, :remx] = img[height-remy:height, xinit:width]
                parts.append(border.astype('float32') / 255)

        return np.asarray(parts), (img.shape[0], img.shape[1])

    def prepare_image_predict(self, input_image):
        original_model_size = IMG_MODEL_SIZE
        downsampling_layer = 4
        border_size = 2**downsampling_layer
        patch_size = IMG_MODEL_SIZE - 2*border_size

        img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        width = img.shape[1]
        height = img.shape[0]
        delta_x = width // patch_size
        delta_y = height // patch_size
        remx = width % patch_size
        remy = height % patch_size

        print(width, patch_size, delta_x, remx)
        print(height, patch_size, delta_y, remy)

        border_width = border_size + width + patch_size - remx + border_size
        border_height = border_size + height  + patch_size - remy + border_size

        parts = []
        border = np.zeros([border_height, border_width], dtype=np.uint8)
        border.fill(255)  # or img[:] = 255

        border[border_size:height+border_size, border_size:width+border_size] = img
        #cv2.imwrite(os.path.join('results', 'img.png'), img)
        #cv2.imwrite(os.path.join('results', 'border.png'), border)

        if remx > 0:
            delta_x += 1
        if remy > 0:
            delta_y += 1

        for x in range(delta_x):
            xinit = x * patch_size
            for y in range(delta_y):
                yinit = y * patch_size
                #print("ystart: " + str(yinit) + ", yend: " + str(yinit+original_model_size) + ", imgstart: " + str(yinit+border_size) + ", imgend: " + str(yinit+original_model_size-border_size))
                part = border[yinit:yinit + original_model_size, xinit:xinit+original_model_size]
                parts.append(part.astype('float32') / 255)

        return np.asarray(parts), (img.shape[0], img.shape[1])


    def restore_image_original(self, parts, dim):
        width = dim[1]
        height = dim[0]
        result = np.zeros([height, width, 1], dtype=np.uint8)
        result.fill(255)  # or img[:] = 255
        delta_x = width // IMG_MODEL_SIZE
        delta_y = height // IMG_MODEL_SIZE
        remx = width % IMG_MODEL_SIZE
        remy = height % IMG_MODEL_SIZE
        index = 0
        for x in range(delta_x):
            xinit = x * IMG_MODEL_SIZE
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                result[yinit:yinit+IMG_MODEL_SIZE, xinit:xinit+IMG_MODEL_SIZE] = parts[index] * 255
                index += 1
            if remy > 0:
                result[height-remy:, xinit:xinit+IMG_MODEL_SIZE] = parts[index][:remy, :] * 255
                index += 1
        if remx > 0:
            xinit = width - remx
            for y in range(delta_y):
                yinit = y * IMG_MODEL_SIZE
                result[yinit:yinit+IMG_MODEL_SIZE, xinit:] = parts[index][:, :remx] * 255
                index += 1
            if remy > 0:
                result[height-remy:, xinit:xinit+IMG_MODEL_SIZE] = parts[index][:remy, :remx] * 255

        return result

    def restore_image(self, parts, dim):
        original_model_size = IMG_MODEL_SIZE
        downsampling_layer = 4
        border_size = 2**downsampling_layer
        patch_size = IMG_MODEL_SIZE - 2*border_size

        width = dim[1] # original image size
        height = dim[0]
        result = np.zeros([height, width, 1], dtype=np.float32)
        result.fill(255)  # or img[:] = 255
        delta_x = width // patch_size
        delta_y = height // patch_size
        remx = width % patch_size
        remy = height % patch_size
        index = 0

        #print(width, delta_x, remx)
        #print(height, delta_y, remy)
        if remx > 0:
            delta_x += 1

        for x in range(delta_x):
            xinit = x * patch_size
            for y in range(delta_y):
                yinit = y * patch_size
                part = parts[index]
                if x == delta_x-1 and remx > 0: # right of picture (remx)
                    result[yinit:yinit+patch_size, xinit:xinit+remx] = 255 * part[border_size:border_size+patch_size, border_size:border_size+remx]
                    #print(255 * part[border_size:border_size+patch_size, border_size:border_size+remx])
                else: # Standard case, no remx or remy problems
                    result[yinit:yinit+patch_size, xinit:xinit+patch_size] = 255 * part[border_size:border_size+patch_size, border_size:border_size+patch_size]
                    #print(255 * part[border_size:border_size+patch_size, border_size:border_size+patch_size])
                index += 1

            if remy > 0 and x == delta_x-1 and remx > 0: # bottom-right of picture (remy + remx)
                part = part = parts[index]
                result[height-remy:height, width-remx:width] = 255 * part[border_size:border_size+remy, border_size:border_size+remx]
                #print(255 * part[border_size:border_size+remy, border_size:border_size+remx])
                index += 1
            elif remy > 0: # bottom of picture (remy)
                part = part = parts[index]
                result[height-remy:height, xinit:xinit+patch_size] = 255 * part[border_size:border_size+remy, border_size:border_size+patch_size]
                #print(255 * part[border_size:border_size+remy, border_size:border_size+patch_size])
                index += 1
        return result


    def binarise_image(self, input_image, name, model_weights=None, model=None):
        print("loading image")
        parts, dim = self.prepare_image_predict(input_image=input_image)

        if model_weights is not None:
            model = self.get_unet()
            model.load_weights(model_weights)

        print('predicting test data')
        imgs_mask_test = model.predict(parts, batch_size=1, verbose=1)

        # for i, predicted_part in enumerate(imgs_mask_test[:,:,:,0]):
        #     cv2.imwrite(os.path.join('results', 'predicted_' + str(i) + '.png'), predicted_part*255)
        #     print("Saved predicted part ", i)
        
        # BUILD/RESTORE PREDICTED IMAGE FROM PREDICTED PARTS
        neg_e = self.restore_image(imgs_mask_test, dim)
        cv2.imwrite(os.path.join('results', f'neg_e_predicted_{name}.png'), neg_e)

        x = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

        neg_e = neg_e[:,:,0]
        xu = image_add(neg_e, x)

        #print(neg_e)
        #print(xu)

        # #awawdawd
        # mask = cv2.imread(os.path.join('images', 'GT', name+'.png'), cv2.IMREAD_GRAYSCALE)

        # mask = ((mask/255 > 0.5)).astype(np.uint8)
        # x = x/255
        # x = np.expand_dims(x, axis=2)
        # grayscale_gt = gt_construction.generate_grayscale_gt(x, np.expand_dims(mask, axis=2))

        # neg_e_real = grayscale_gt-x # grayscale_gt-img
        # neg_e_real = remove_negative_pixels(neg_e_real)
        # neg_e_real = (neg_e_real * 255).astype(np.uint8)

        # cv2.imwrite(os.path.join('results', name + '_XU_REAL.png'), image_add(neg_e_real, x*255))
        #cv2.imwrite(os.path.join('results', name + '_NEG_E_PREDICTED.png'), neg_e)

        return xu



    def save_epoch_results(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        model.load_weights('unet_dibco.hdf5')

        print('predict test data')
        imgs = model.predict(imgs_test, batch_size=1, verbose=1)
        path = "D:\\Roe\\Medium\\prjs\\u_net\\results\\" + str(self.counter)

        self.counter += 1
        if not os.path.exists(path):
            makedirs(path)

        for i in range(imgs.shape[0]):
            img = imgs[i]
            # self.show_image(img)
            img = array_to_img(img)
            img.save(path + "%d.jpg" % (i))


    def show_image(self, image, *args, **kwargs):
        title = kwargs.get('title', 'Figure')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        plt.imshow(image)
        plt.title(title)
        plt.show()


def test_predict(u_net, models):
    images_path = os.path.join('images', 'Originals')
    print(images_path)
    images = os.listdir(images_path)
    print(images)
    results = []
    for image in images:
        gt_path = os.path.join('images', 'GT', image[:-4] + '.png')
        # print(gt_path)
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        current_image = os.path.join('images', 'Originals', image)
        image_read = cv2.imread(current_image, cv2.IMREAD_GRAYSCALE)
        for i,model in enumerate(models):
            unet_image = u_net.binarise_image(model_weights=model, input_image=current_image, name=image[:-4])
            current_image = os.path.join('temp_stack',image[:-4] + '_' + str(i) +'.png')
            cv2.imwrite(current_image,unet_image)
            
        result_unet = unet_image
        ressult_otsu = threshold_otsu(result_unet)
        cv2.imwrite(os.path.join('results', image[:-4] + '_' + '_unet_.png'), result_unet)
        # Binarise image with Otsu
        result_unet = ((result_unet > ressult_otsu) * 255).astype(np.uint8)
        ressult_otsu = threshold_otsu(image_read)
        result_otsu = ((image_read > ressult_otsu) * 255).astype(np.uint8)
        result_sauvola = threshold_sauvola(image_read)
        result_sauvola = ((image_read > result_sauvola) * 255).astype(np.uint8)
        window_size = 25
        result_niblack = threshold_niblack(image_read, window_size=window_size, k=0.8)
        result_niblack = ((image_read > result_niblack) * 255).astype(np.uint8)

        #print(result_unet.shape)
        #print("------")
        #print(result_niblack.shape)
        #exit()


        # FIX ground_truth and output from unet to be binarized (0 or 255)
        ground_truth = ((ground_truth > 128) * 255).astype(np.uint8)
        #result_unet = result_unet[:,:,0]
        #result_unet = ((result_unet > 128) * 255).astype(np.uint8)
        #for pixel in result_unet:
        #    print(pixel)

        img_true = np.array(ground_truth).ravel()
        img_pred = np.array(result_niblack).ravel()
        iou_niblack = statistics.mean(jaccard_score(img_true, img_pred, average=None))
        cv2.imwrite(os.path.join('results', image[:-4] + '_' + str(iou_niblack)[:5] + '_niblack_.png'),
                    result_niblack)

        img_pred = np.array(result_otsu).ravel()
        iou_otsu = statistics.mean(jaccard_score(img_true, img_pred, average=None))
        cv2.imwrite(os.path.join('results', image[:-4] + '_' + str(iou_otsu)[:5] + '_otsu_.png'), result_otsu)

        img_pred = np.array(result_sauvola).ravel()
        iou_sauvola = statistics.mean(jaccard_score(img_true, img_pred, average=None))
        cv2.imwrite(os.path.join('results', image[:-4] + '_' + str(iou_sauvola)[:5] + '_sauvola_.png'),
                    result_sauvola)

        img_pred = np.array(result_unet).ravel()
        print("img_true")
        print(img_true)
        print("img_pred")
        print(img_pred)
        iou_unet = statistics.mean(jaccard_score(img_true, img_pred, average=None))
        cv2.imwrite(os.path.join('results', image[:-4] + '_' + str(iou_unet)[:5] + '_unet_binarize.png'), result_unet)

        results.append([iou_unet, iou_otsu, iou_sauvola, iou_niblack])

    for index, image in enumerate(images):
        print('Image', image, '- U-Net IoU:', results[index][0], 'Otsu IoU:', results[index][1], 'Sauvola IoU:',
              results[index][2], 'Niblack IoU:', results[index][3])


if __name__ == '__main__':
    check_gpu()
    my_unet = myUnet()

    #data_path = os.path.join('..', 'destination')
    #checkpoint_file = '..//model//unet_testing_dataset.hdf5'
    #my_unet.train(data_path, checkpoint_file, epochs=5)

    #If you want to test the model just uncomment the following code
    #Pre-trained model
    model = os.path.join('..', 'model', 'unet_testing_dataset.hdf5')
    test_predict(my_unet, model)