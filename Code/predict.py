import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import ImageFilter
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error


def predict(job_dir="..", file_name="/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg"):
    gen = load_model(job_dir + "/checkpoint/generator/")
    img = np.array(Image.open(job_dir + file_name).resize((200, 200)))
    source_age = int(file_name.split("/")[2].split("_")[0])
    print("Source age = " + str(source_age))
    target_age = int(input("Enter the target age: "))

    def age_group(age):
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

    img = img[None, :, :, :]
    target_age_group = np.array(tf.one_hot(age_group(target_age), 14))[None, :]
    print(target_age_group.shape)

    _, output_image = gen([img, target_age_group])
    output_image = np.rint(output_image.numpy()[0]).astype(int)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    dis = load_model(job_dir + "/checkpoint/discriminator/")
    prediction = dis(output_image[None, :, :, :])
    print(prediction)

    source_age_group = np.array(tf.one_hot(age_group(source_age), 14))[None, :]
    _, output_image_1 = gen([output_image[None, :, :, :], source_age_group])
    output_image_1 = np.rint(output_image_1.numpy()[0]).astype(int)
    output_image_1 = np.clip(output_image_1, 0, 255).astype(np.uint8)

    reconstruction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                Flatten()(img[0].astype(float)),
                Flatten()(output_image_1.astype(float)))))

    print(reconstruction_loss)

    im = Image.fromarray(output_image).filter(ImageFilter.SMOOTH_MORE)
    im1 = Image.fromarray(output_image_1).filter(ImageFilter.SMOOTH_MORE)
    im.save(job_dir + "/Predictions/Result_target.jpg")
    im1.save(job_dir + "/Predictions/Result_source.jpg")


if __name__ == 'main':
    predict()
    
    
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import ImageFilter
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error


def predict(job_dir="..", file_name="/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg"):
    if not os.path.exists(job_dir + "/checkpoint/generator/"):
        print("Model not found")
        exit()
    gen = load_model(job_dir + "/checkpoint/generator/")
    img = np.array(Image.open(job_dir + file_name).resize((200, 200)))
    source_age = int(file_name.split("/")[2].split("_")[0])
    print("Source age = " + str(source_age))
    target_age = int(input("Enter the target age: "))

    if target_age <= 5:
        target_age_group = 0
    elif target_age <= 10:
        target_age_group = 1
    elif target_age <= 15:
        target_age_group = 2
    elif target_age <= 20:
        target_age_group = 3
    elif target_age <= 25:
        target_age_group = 4
    elif target_age <= 30:
        target_age_group = 5
    elif target_age <= 40:
        target_age_group = 6
    elif target_age <= 50:
        target_age_group = 7
    elif target_age <= 60:
        target_age_group = 8
    elif target_age <= 70:
        target_age_group = 9
    elif target_age <= 80:
        target_age_group = 10
    elif target_age <= 90:
        target_age_group = 11
    elif target_age_group <= 100:
        target_age_group = 12
    else:
        target_age_group = 13

    img = img[None, :, :, :]
    target_age_group = np.array([target_age_group])[None, :]

    _, output_image = gen([img, target_age_group])
    output_image = np.rint(output_image.numpy()[0]).astype(int)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    dis = load_model(job_dir + "/checkpoint/discriminator/")
    prediction = dis(output_image[None, :, :, :])
    print(prediction)
    
    print(tf.sqrt(
        mean_squared_error(
            target_age_group,
            prediction)))

    _, output_image_1 = gen([output_image[None, :, :, :], np.array([7])[None, :]])
    output_image_1 = np.rint(output_image_1.numpy()[0]).astype(int)
    output_image_1 = np.clip(output_image_1, 0, 255).astype(np.uint8)

    reconstruction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                Flatten()(img[0].astype(float)),
                Flatten()(output_image_1.astype(float)))))

    print(reconstruction_loss)
    
    prediction = dis(output_image_1[None, :, :, :])
    print(prediction)
    
    print(tf.sqrt(
        mean_squared_error(
            np.array([7])[None, :],
            prediction)))
    
    im = Image.fromarray(output_image).filter(ImageFilter.SMOOTH_MORE)
    im1 = Image.fromarray(output_image_1).filter(ImageFilter.SMOOTH_MORE)
    im.save(job_dir + "/Result.jpg")
    im1.save(job_dir + "/Result1.jpg")


if __name__ == "__main__":
    predict()

"""
