import numpy
import keras
import os
import tensorflow as tf

def images_to_array(dataset_dir, image_size):
    dataset_array = []
    dataset_labels = []

    class_counter = 0

    classes_names = os.listdir(dataset_dir)
    for current_class_name in classes_names:
        class_dir = os.path.join(dataset_dir, current_class_name)
        images_in_class = os.listdir(class_dir)

        print("Class index", class_counter, ", ", current_class_name, ":" , len(images_in_class))

        for image_file in images_in_class:
            if image_file.endswith(".jpg"):
              image_file_dir = os.path.join(class_dir, image_file)

              img = keras.preprocessing.image.load_img(image_file_dir, target_size=(image_size, image_size))
              img_array = keras.preprocessing.image.img_to_array(img)

              img_array = img_array/255.0

              dataset_array.append(img_array)
              dataset_labels.append(class_counter)
        class_counter = class_counter + 1
    dataset_array = numpy.array(dataset_array)
    dataset_labels = numpy.array(dataset_labels)
    return dataset_array, dataset_labels

zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/kaggle-datasets/5857/414958/fruits-360_dataset.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1566314394&Signature=rqm4aS%2FIxgWrNSJ84ccMfQGpjJzZ7gZ9WmsKok6quQMyKin14vJyHBMSjqXSHR%2B6isrN2POYfwXqAgibIkeAy%2FcB2OIrxTsBtCmKUQKuzqdGMnXiLUiSKw0XgKUfvYOTC%2F0gdWMv%2B2xMLMjZQ3CYwUHNnPoDRPm9GopyyA6kZO%2B0UpwB59uwhADNiDNdVgD3GPMVleo4hPdOBVHpaWl%2F%2B%2BPDkOmQdlcH6b%2F983JHaktssmnCu8f0LVeQjzZY96d24O4H85x8wdZtmkHZCoFiIgCCMU%2BKMMBAbTL66QiUUB%2FW%2FpULPlpzN9sBBUR2yydB3CUwqLmSjAcwz3wQ%2FpIhzg%3D%3D",
                                   fname="fruits-360.zip", extract=True)
base_dir, _ = os.path.splitext(zip_file)

train_dir = "/root/.keras/datasets/fruits-360/Training"
image_size = 128
train_dataset_array, train_dataset_array_labels = images_to_array(dataset_dir=train_dir, image_size=image_size)
print("Training Data Array Shape :", train_dataset_array.shape)
numpy.save("train_dataset_array.npy", train_dataset_array)
numpy.save("train_dataset_array_labels.npy", train_dataset_array_labels)

test_dir = "/root/.keras/datasets/fruits-360/Test"
test_dataset_array, test_dataset_array_labels = images_to_array(dataset_dir=test_dir, image_size=image_size)
print("Test Data Array Shape :", test_dataset_array.shape)
numpy.save("test_dataset_array.npy", test_dataset_array)
numpy.save("test_dataset_array_labels.npy", test_dataset_array_labels)