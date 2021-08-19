# Facial Landmarks Detection and Extraction
import os

from face_landmarks_detection import initialize_face_detection, process_input_image, face_detector
from file_handler import dataset_path, shape_predictor_path, get_file_path, is_valid_file, clear_results


# initialize the project
def initialize():
    # get Dlib shape detector and predictor
    [detector, predictor] = initialize_face_detection(shape_predictor_path)

    # loop over images in dataset
    for file_name in os.listdir(dataset_path):
        if is_valid_file(dataset_path + file_name, file_name):
            [image_path, image_segments_dir] = get_file_path(file_name)
            [resized_image, gray_image] = process_input_image(image_path)
            face_detector(detector, predictor, resized_image, gray_image, image_segments_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clear_results()
    initialize()
