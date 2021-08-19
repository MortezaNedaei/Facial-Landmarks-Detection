# Facial Landmarks Detection and Extraction

from face_landmarks_detection import initialize_face_detection, setup_dataset, face_detector

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
dataset_path = "dataset/image_1.jpg"


def initialize():
    [detector, predictor] = initialize_face_detection(shape_predictor_path)
    [resized_image, gray_image] = setup_dataset(dataset_path)
    face_detector(detector, predictor, resized_image, gray_image)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initialize()
