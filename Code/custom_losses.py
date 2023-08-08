"""Defines the custom loss functions used by the GAN model"""

import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.losses import categorical_crossentropy


def gen_id_loss_fn(identity_1, identity_2):
	# Defines the identity loss function
	identity_loss = tf.sqrt(mean_squared_error(Flatten()(identity_1), Flatten()(identity_2)))
	return identity_loss


def real_age_loss_fn(output, target):
	# Defines the real age loss function
	age_prediction_loss = categorical_crossentropy(target, output)
	return age_prediction_loss


def gen_img_loss_fn(input_image, output_image_2):
	# Defines the image comparison loss function
	reconstruction_loss = mean_absolute_error(Flatten()(input_image), Flatten()(output_image_2))
	return reconstruction_loss


def fake_age_loss_fn(output, target):
	# Defines the fake age loss function
	age_prediction_loss = mean_absolute_error(target, output)
	return age_prediction_loss
