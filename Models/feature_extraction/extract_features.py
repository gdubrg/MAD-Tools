from os.path import join, dirname, basename
from glob import glob
from tqdm import tqdm
import argparse
import face_model
import cv2
import os
import numpy as np
import sys

# Path to ArcFace model and Face Detector (MTCNN) model
path_model = './model,0000'
path_mtcnn = './mtcnn-model'

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default=path_model, help='path to load model.')
parser.add_argument('--modelmtcnn', default=path_mtcnn)
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--start', default=0, type=int, help='start')
parser.add_argument('--end', default=0, type=int, help='end')
args = parser.parse_args()


def extract_features():

    # save face detection images
    SAVE_IMAGES = True

    # save extracted features
    SAVE_FEATURES = False

    # input path
    path_dataset = ''

    # output path
    path_output = ''

    # output path for images
    path_output_images = ''

    # IMPORTANT: here change code according to input directory tree
    images = sorted(glob(join(path_dataset, "*", "*.png")))

    model = face_model.FaceModel(args)

    for image in tqdm(images):

        output_name = image.replace(path_dataset, path_output)
        output_name_images = image.replace(path_dataset, path_output_images)
        if not os.path.exists(dirname(output_name)):
            os.makedirs(dirname(output_name))

        img = cv2.imread(image)
        img = model.get_input(img)

        if SAVE_IMAGES:
            if not os.path.exists(dirname(output_name_images)):
                os.makedirs(dirname(output_name_images))
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_name_images, img)

        if SAVE_FEATURES:
            f1 = model.get_feature(img)
            np.save(output_name[:-4], f1)


if __name__ == "__main__":
    extract_features()