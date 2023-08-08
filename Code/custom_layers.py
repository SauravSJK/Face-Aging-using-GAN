"""Defines the custom layers and functions used for by the generator and discriminator models"""

import tensorflow as tf
from tensorflow.nn import moments
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.nn import batch_normalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import Ones
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GroupNormalization


class ConditionBatchNormalization(Layer):
	# Implements the conditional batch normalization layer
	# Ref: https://colab.research.google.com/drive/1WGG8d22KoxXWBThYOeFDcHvt_z9EirHV#scrollTo=-CxyRhZaDSYk
	def __init__(self):
		# Initializes the layer variables
		super(ConditionBatchNormalization, self).__init__()
		self.decay = 0.9
		self.epsilon = 1e-05
		self.test_var = None
		self.test_mean = None
		self.num_channels = None
		self.beta_mapping = None
		self.gamma_mapping = None

	def build(self, input_shape):
		# Modifies the variables based on the input data shape
		self.num_channels = input_shape[0][-1]
		self.beta_mapping = Dense(self.num_channels, kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
		                          activity_regularizer='l1_l2')
		self.gamma_mapping = Dense(self.num_channels, kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
		                           activity_regularizer='l1_l2')
		self.test_mean = tf.Variable(initial_value=Zeros()(self.num_channels), trainable=False, dtype=tf.float32)
		self.test_var = tf.Variable(initial_value=Ones()(self.num_channels), trainable=False, dtype=tf.float32)

	def call(self, x, training=None):
		# Defines the calculations to be performed by the layer
		x, conditions = x
		beta = self.beta_mapping(conditions)
		gamma = self.gamma_mapping(conditions)
		beta = tf.reshape(beta, shape=[-1, 1, 1, self.num_channels])
		gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.num_channels])
		if training:
			# Calculate mean and variance of X
			batch_mean, batch_var = moments(x, [0, 1, 2])
			# Calculate parameters for test set
			test_mean = self.test_mean * self.decay + batch_mean * (1 - self.decay)
			test_var = self.test_var * self.decay + batch_var * (1 - self.decay)

			def mean_update():
				self.test_mean.assign(test_mean)

			def variance_update():
				self.test_var.assign(test_var)

			self.add_update(mean_update)
			self.add_update(variance_update)

			return batch_normalization(x, batch_mean, batch_var, beta, gamma, self.epsilon)
		else:
			return batch_normalization(x, self.test_mean, self.test_var, beta, gamma, self.epsilon)

	def get_config(self):
		# Returns the configuration of the layer
		config = super().get_config()
		config.update({"units": self.num_channels})
		return config


def ada_in(style, content, epsilon=1e-5):
	# Implements adaptive instance normalization
	# Ref: https://github.com/ariG23498/AdaIN-TF/blob/master/AdaIN.ipynb
	axes = [1, 2]

	c_mean, c_var = moments(content, axes=axes, keepdims=True)
	s_mean, s_var = moments(style, axes=axes, keepdims=True)
	c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

	t = s_std * (content - c_mean) / c_std + s_mean
	return t


def res_block(x, filters, kernel_size=4, sampling=None, sampling_size=2, norm_type=None, condition=None):
	# Defines the residual block unit used by the generator and discriminator models
	if norm_type == "instance":
		y = GroupNormalization(groups=-1)(x)
	elif norm_type == "cbn":
		y = ConditionBatchNormalization()([x, condition])
	elif norm_type == "ada_in":
		condition = Conv2D(x.shape[3], 1, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
		                   activity_regularizer='l1_l2')(condition)
		y = ada_in(condition, x)
	else:
		y = x
	y = LeakyReLU()(y)
	y = Conv2D(filters, kernel_size, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
	           activity_regularizer='l1_l2')(y)
	if norm_type == "instance":
		#y = InstanceNormalization()(y)
		y = GroupNormalization(groups=-1)(y)
	elif norm_type == "cbn":
		y = ConditionBatchNormalization()([y, condition])
	elif norm_type == "ada_in":
		condition = Conv2D(y.shape[3], 1, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
		                   activity_regularizer='l1_l2')(condition)
		y = ada_in(condition, y)
	else:
		y = y
	y = LeakyReLU()(y)
	y = Conv2D(filters, kernel_size, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
	           activity_regularizer='l1_l2')(y)
	z = Conv2D(filters, 1, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2',
	           activity_regularizer='l1_l2')(x)
	out = y + z
	if sampling == "down":
		return AveragePooling2D(pool_size=sampling_size)(out)
	elif sampling == "up":
		return UpSampling2D(size=sampling_size)(out)
	else:
		return out