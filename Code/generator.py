"""Defines the generator model"""

from custom_layers import res_block
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D


def encoder():
	# Defines the encoder sub-model
	input_image = Input(shape=(200, 200, 3))
	x = Conv2D(filters=64, kernel_size=4, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
	           activity_regularizer='l1_l2')(input_image)
	x = res_block(x, 128, sampling="down", norm_type="instance")
	x = res_block(x, 256, sampling="down", norm_type="instance")
	x = res_block(x, 512, sampling="down", norm_type="instance")
	x = res_block(x, 512, sampling="down", sampling_size=5, norm_type="instance")
	x = res_block(x, 512)
	id_ = res_block(x, 512)
	return input_image, id_


def identity(id_):
	# Defines the identity modification sub-model
	age = Input(shape=(14))
	x = res_block(id_, 512, sampling="down", norm_type="cbn", condition=age)
	f_age_aware = Conv2D(filters=512, kernel_size=2, padding="valid", kernel_regularizer='l1_l2',
	                     bias_regularizer='l1_l2', activity_regularizer='l1_l2')(x)
	return age, f_age_aware


def decoder(id_, f_age_aware):
	# Defines the decoder sub-model
	x = res_block(id_, 512, norm_type="ada_in", condition=f_age_aware)
	x = res_block(x, 512, norm_type="ada_in", condition=f_age_aware)
	x = res_block(x, 512, sampling="up", sampling_size=5, norm_type="ada_in", condition=f_age_aware)
	x = res_block(x, 256, sampling="up", norm_type="ada_in", condition=f_age_aware)
	x = res_block(x, 128, sampling="up", norm_type="ada_in", condition=f_age_aware)
	x = res_block(x, 64, sampling="up", norm_type="ada_in", condition=f_age_aware)
	output_image = Conv2D(3, 1, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
	                      activity_regularizer='l1_l2', activation='tanh')(x)
	return output_image


def generator(name=None):
	# Combines the sub-models to form the generator model
	input_image, id_ = encoder()
	age, f_age_aware = identity(id_)
	output_image = decoder(id_, f_age_aware)
	gen = Model(inputs=[input_image, age], outputs=[id_, output_image], name=name)
	return gen
