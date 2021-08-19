import os
import shutil

import cv2

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
dataset_path = "dataset/"


def get_file_path(file_name_with_extension):
    image_path = os.path.join(dataset_path, file_name_with_extension)  # dataset/image_1.jpg
    image_segments_dir = os.path.splitext(image_path)[0] + '/'  # dataset/image_1/
    os.makedirs(image_segments_dir, exist_ok=True)
    return [image_path, image_segments_dir]


def write_file(file_name, file):
    cv2.imwrite(file_name + '.jpg', file)


# checks if path is a file
def is_file(path):
    return os.path.isfile(path)


# checks if path is a directory
def is_directory(path):
    return os.path.isdir(path)


# checks if path is a valid file
def is_valid_file(path, file_name):
    return is_file(path) and file_name != ".DS_Store"


# remove all results
def clear_results():
    for file_name in os.listdir(dataset_path):
        if is_directory(dataset_path + file_name):
            shutil.rmtree(dataset_path + file_name)
