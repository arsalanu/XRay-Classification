import os
import json
import cv2
import numpy as np

def label_generator(label_file):
    with open(label_file, "rb+") as label_data:
        label_json = json.load(label_data)

    return [(label["class_name"], int(label["id"])) for label in label_json["labels"]]


def extract_image_data(labels, image_dir, image_input_size):
    image_data = []

    for label in labels:
        file_path = os.path.join(image_dir, label[0])

        for file in os.listdir(file_path):
            image = cv2.imread(os.path.join(file_path, file))
            image = cv2.resize(image, (image_input_size[0], image_input_size[1]))
            image_data.append([image, int(label[1])])

    return np.array(image_data)


def reshape_normalise_data(data, image_input_size):
    x = []
    y = []

    for f,l in data:
        x.append(f)
        y.append(l)

    x = np.array(x) / 255
    x.reshape(-1, image_input_size[0], image_input_size[1], 1)

    y = np.array(y)
    return x, y


def extract(request):
    label_file_path = "labeldata.json"  # json which has the label data for images
    image_dir_path = "Datasets/archive/chest_xray"  # Where test, train and validation images are stored
    image_input_size = [128, 128]  # Images resized to a uniform size for model fitting
    output_string = ""
    request_dict = {}

    if "train" in request:
        train_data = extract_image_data(
            label_generator(label_file_path),
            os.path.join(image_dir_path, "train"),
            image_input_size)

        x_train, y_train = reshape_normalise_data(train_data, image_input_size)

        request_dict["x_train"] = x_train
        request_dict["y_train"] = y_train

        output_string += "Returned train data \n"

    if "test" in request:
        test_data = extract_image_data(
            label_generator(label_file_path),
            os.path.join(image_dir_path, "test"),
            image_input_size)

        x_test, y_test = reshape_normalise_data(test_data,image_input_size)

        request_dict["x_test"] = x_test
        request_dict["y_test"] = y_test

        output_string += "Returned test data \n"

    if "val" in request:
        val_data = extract_image_data(
            label_generator(label_file_path),
            os.path.join(image_dir_path, "val"),
            image_input_size)

        x_val, y_val = reshape_normalise_data(val_data,image_input_size)

        request_dict["x_val"] = x_val
        request_dict["y_val"] = y_val

        output_string += "Returned validation data \n"

    if output_string == "":
        output_string = "[E] Nothing returned, use \"train\", \"test\" or \"val\" in request to get data"

    return request_dict, output_string, image_input_size
