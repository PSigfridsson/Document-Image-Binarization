
import os
import numpy as np
import cv2
from PIL import Image
from skimage.util import random_noise

IMG_MODEL_SIZE = 128


def rotate_img(img, rt_degr):
    img = Image.fromarray(img)

    return np.asarray(img.rotate(rt_degr, expand=1))


def rotate():
   ...


def invert(image):
    return 255 - image


def cut_image(img, step):
    # print('cutting', img.shape)
    width = img.shape[1]
    height = img.shape[0]
    delta_x = step
    delta_y = step
    cuts = []
    curr_x = 0
    curr_y = 0
    # step = 128
    first = True
    while curr_x + IMG_MODEL_SIZE < width:
        curr_y = 0
        while curr_y + IMG_MODEL_SIZE < height:
            cuts.append(img[curr_y:curr_y + IMG_MODEL_SIZE , curr_x:curr_x + IMG_MODEL_SIZE])
            
            curr_y += step
        curr_x += step
    return cuts


def myFunc(img):
    print('This is another function.')

def myFunc2():
    print('This is another function2.')


def noise(image):
    amount = 0.01
    res = random_noise(image, mode='s&p', amount=amount)

    return np.array(255 * res, dtype='uint8')


def create_augmentation(root_files_path, destination_path):

    # functions = {'myfoo2': myFunc2, 'myfoo': myFunc}
    # for name in functions.keys():
    #     res = functions[name](image_src)

    cut_step = IMG_MODEL_SIZE # 50 -> 241.731 100 -> 64.169
    datasets = os.listdir(root_files_path)
    print(datasets)
    for dataset in datasets:
        if dataset in ['Bickley','nabuco-dataset-1','ektaBinExp','Cuper','irish', 'ICDAR']:
            continue

        # if dataset not in ['dibco']:
        #     continue
        
        sub_datas = os.listdir(os.path.join(root_files_path, dataset))
        print(sub_datas)
        for sub_data in sub_datas:
            originals_path = 'Originals'
            gt_path = 'GT'

            files = os.listdir(os.path.join(root_files_path, dataset, sub_data, originals_path))
            for file in files:
                print(os.path.join(root_files_path, dataset, sub_data, originals_path, file))
                image_original = cv2.imread(os.path.join(root_files_path, dataset, sub_data, originals_path, file))
                image_gt = cv2.imread(os.path.join(root_files_path, dataset, sub_data, gt_path, file[:-4] + '_gt.png'))

                images_augmentation_original = []
                images_augmentation_gt = []
                
                names = ['_orig']
                images_augmentation_original.append(image_original)
                images_augmentation_gt.append(image_gt)
                
                #images_augmentation_original.append(cv2.flip(image_original, 0))
                #images_augmentation_gt.append(cv2.flip(image_gt, 0))
                #names.append('_fliph')
                #images_augmentation_original.append(cv2.flip(image_original, 1))
                #images_augmentation_gt.append(cv2.flip(image_gt, 1))
                #names.append('_flipv')

                #images_augmentation_original.append(cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY))
                #images_augmentation_gt.append(image_gt)
                #names.append('_grey')

                #images_augmentation_original.append(noise(image_original))
                #images_augmentation_gt.append(image_gt)
                #names.append('_nois')
                
                
                #for ang in [90, 180, 270]:
                 #   names.append('_ang'+str(ang))
                  #  images_augmentation_original.append(rotate_img(image_original, ang))
                   # images_augmentation_gt.append(rotate_img(image_gt, ang))                

                print(names, len(images_augmentation_original))

                for index in range(len(images_augmentation_original)):

                    name_augm = names[index]
                    cuts_originals = cut_image(images_augmentation_original[index], cut_step)
                    cuts_gt = cut_image(images_augmentation_gt[index], cut_step)
                    for i,cut in enumerate(cuts_originals):
                          name = file[:-4] + name_augm + '_' + str(i) + '.png'
                          print(name)
                          if cut.shape != (128,128,3):
                              print("not the right size lul")
                              exit()
                          cv2.imwrite(os.path.join(destination_path, 'Originals', name), cut)

                    for i,cut in enumerate(cuts_gt):
                          name = file[:-4] + name_augm + '_' + str(i) + '.png'
                          print(name)
                          if cut.shape != (128,128,3):
                              print("not the right size lul")
                              exit()
                          cv2.imwrite(os.path.join(destination_path, 'GT', name), cut)
                          
                   
def convert(root_path):
    datasets = os.listdir(root_path)
    tot_all = 0
    for dataset in datasets:
        if dataset in ['nop', 'backup']:
            continue
        print(dataset)
        sub_datas = os.listdir(os.path.join(root_path, dataset))
        for subdata in sub_datas:
            print('   ', subdata)
            gts = os.listdir(os.path.join(root_path, dataset, subdata, 'GT'))
            for file in gts:
                if 'png' not in file:
                    print('         ', file)
                    tot_all += 1
                    delt = 3
                    if 'tiff' in file:
                        delt = 4
                    image = cv2.imread(os.path.join(root_path, dataset, subdata, 'GT', file))
                    cv2.imwrite(os.path.join(root_path, dataset, subdata, 'GT', file[:-delt]+'png'), image)
                    print('writing', os.path.join(root_path, dataset, subdata, 'GT', file[:-delt]+'png'))
                    os.remove(os.path.join(root_path, dataset, subdata, 'GT', file))
    print(tot_all)

def check_files(root_path):
    # dibco 116
    # ICDAR 4782
    # ICFHR_2016 100
    # nabuco-dataset 15
    # PHIB 15
    # Total 5028
    datasets = os.listdir(root_path)
    tot_all = 0
    for dataset in datasets:
        if dataset in ['nop', 'backup', 'augmentations']:
            continue
        tot_dataset = 0
        sub_datas = os.listdir(os.path.join(root_path, dataset))
        for subdata in sub_datas:
            originals = os.listdir(os.path.join(root_path, dataset, subdata, 'Originals'))
            for original in originals:
                name_gt = original[:-3] + 'png'
                if not os.path.exists(os.path.join(root_path, dataset, subdata, 'GT', name_gt)):
                    print('error', original, name_gt)
        tot_all += tot_dataset
        print(dataset, tot_dataset)
    print(tot_all)



def rename_dataset(root_path):
    # dibco 116
    # ICDAR 4782
    # ICFHR_2016 100
    # nabuco-dataset 15
    # PHIB 15
    # Total 5028
    datasets = os.listdir(root_path)
    tot_all = 0
    for dataset in datasets:
        originals = os.listdir(os.path.join(root_path, 'Originals'))
        gt = os.listdir(os.path.join(root_path, 'GT'))
        for index in range(len(gt)):
            if originals[index] != gt[index]:
                print(originals[index], '---', gt[index])
                os.rename(os.path.join(root_path, 'GT', gt[index]),
                          os.path.join(root_path, 'GT', originals[index][:-3]+'png'))


import sys
if __name__ == '__main__':
    #root = 'D:\\Roe\\Medium\\data'
    testdest = os.path.join('..','..','destination')
    testfiles = os.path.join('..','..', 'datasets')
    create_augmentation(testfiles, testdest)
    #img = cv2.imread('D:\\Roe\\Medium\\paper_to\\unet\\figs\\63-IMG_MAX_881468.jpg')
    #cut_image(img, 256)
    # rename_dataset('D:\\Roe\\Medium\\data\\ICDAR\\2017')
    # check_files(root)
    # convert(root)
    # sys.exit()
    #create_augmentation(root, 'D:\\Roe\\Medium\\data\\augmentations')
