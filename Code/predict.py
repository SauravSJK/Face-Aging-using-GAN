"""Defines the predictor function for testing the trained model on a single image"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import ImageFilter
from tensorflow.keras.models import load_model

# Set log level to 3 to print only minimum logs to console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict(file_name="/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg", job_dir=".."):
    # Defines a predict function that runs the generator model for a single image
    gen = load_model(f"{job_dir}/checkpoint/generator/generator.keras")
    # Load the image and ensure that its size is (200, 200)
    img = np.array(Image.open(job_dir + file_name).resize((200, 200)))
    # Get the source age from the filename
    source_age = int(file_name.split("/")[2].split("_")[0])
    print("Source age = " + str(source_age))
    # Get the target age from the user
    target_age: int = int(input("Enter the target age: "))

    def age_group(age):
        # Converts the source and target ages to categories
        if age <= 5:
            age_group = 0
        elif age <= 10:
            age_group = 1
        elif age <= 15:
            age_group = 2
        elif age <= 20:
            age_group = 3
        elif age <= 25:
            age_group = 4
        elif age <= 30:
            age_group = 5
        elif age <= 40:
            age_group = 6
        elif age <= 50:
            age_group = 7
        elif age <= 60:
            age_group = 8
        elif age <= 70:
            age_group = 9
        elif age <= 80:
            age_group = 10
        elif age <= 90:
            age_group = 11
        elif age <= 100:
            age_group = 12
        else:
            age_group = 13
        return age_group

    # Convert the image pixel values to the range [-1, 1] and one-hot encode the target age category
    img = img[None, :, :, :] - 127.5 / 127.5
    target_age_group = np.array(tf.one_hot(age_group(target_age), 14))[None, :]

    # Run the generator model on the input image and target age
    # Convert the image pixel values back to the range [0, 255]
    _, output_image = gen([img, target_age_group])
    output_image = np.rint(output_image.numpy()[0] * 127.5 + 127.5).astype(int)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Run the generated image through the discriminator model to get the age prediction
    dis = load_model(job_dir + "/checkpoint/discriminator/")
    prediction = dis(output_image[None, :, :, :])
    print(prediction)

    # Run the generator model on the output image and source age
    # Convert the image pixel values back to the range [0, 255]
    source_age_group = np.array(tf.one_hot(age_group(source_age), 14))[None, :]
    _, output_image_1 = gen([output_image[None, :, :, :] - 127.5 + 127.5, source_age_group])
    output_image_1 = np.rint(output_image_1.numpy()[0] * 127.5 + 127.5).astype(int)
    output_image_1 = np.clip(output_image_1, 0, 255).astype(np.uint8)

    # Save the generated images to disk
    im = Image.fromarray(output_image).filter(ImageFilter.SMOOTH_MORE)
    im1 = Image.fromarray(output_image_1).filter(ImageFilter.SMOOTH_MORE)
    im.save(job_dir + "/Predictions/Result_target.jpg")
    im1.save(job_dir + "/Predictions/Result_source.jpg")