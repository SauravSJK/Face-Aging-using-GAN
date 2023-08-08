"""Defines the custom callback classes used by the GAN model"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class CustomEarlyStopping(Callback):
	"""Implements a custom early stopping callback that uses multiple conditions"""
	# Ref: https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions

	def __init__(self, patience=0):
		# Defines the variables
		super(CustomEarlyStopping, self).__init__()
		self.wait = None
		self.patience = patience
		self.best_weights = None
		self.stopped_epoch = None
		self.best_gen_id_loss_12 = None
		self.best_gen_id_loss_23 = None
		self.best_gen_age_loss_1 = None
		self.best_gen_age_loss_2 = None
		self.best_gen_age_loss_3 = None
		self.best_gen_cycle_loss = None
		self.best_disc_real_loss = None
		self.best_gen_reconn_loss = None
		self.best_disc_fake_loss_1 = None
		self.best_disc_fake_loss_2 = None
		self.best_disc_fake_loss_3 = None

	def on_train_begin(self, logs=None):
		# Initializes the variables at the beginning of the epoch
		# The number of epoch it has waited when loss is no longer minimum.
		self.wait = 0
		# The epoch the training stopped at
		self.stopped_epoch = 0
		# Initialize the best as infinity
		self.best_gen_id_loss_12 = np.Inf
		self.best_gen_id_loss_23 = np.Inf
		self.best_gen_age_loss_1 = np.Inf
		self.best_gen_age_loss_2 = np.Inf
		self.best_gen_age_loss_3 = np.Inf
		self.best_gen_cycle_loss = np.Inf
		self.best_disc_real_loss = np.Inf
		self.best_gen_reconn_loss = np.Inf
		self.best_disc_fake_loss_1 = np.Inf
		self.best_disc_fake_loss_2 = np.Inf
		self.best_disc_fake_loss_3 = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		# Defines the checks for the end of the epoch to decide whether to stop the training
		gen_id_loss_12 = logs.get("val_gen_id_loss_12")
		gen_id_loss_23 = logs.get("val_gen_id_loss_23")
		gen_age_loss_1 = logs.get("val_gen_age_loss_1")
		gen_age_loss_2 = logs.get("val_gen_age_loss_2")
		gen_age_loss_3 = logs.get("val_gen_age_loss_3")
		gen_reconn_loss = logs.get("val_gen_reconn_loss")
		gen_cycle_loss = logs.get("val_gen_cycle_loss")
		disc_real_loss = logs.get("val_disc_real_loss")
		disc_fake_loss_1 = logs.get("val_disc_fake_loss_1")
		disc_fake_loss_2 = logs.get("val_disc_fake_loss_2")
		disc_fake_loss_3 = logs.get("val_disc_fake_loss_3")

		if (np.less_equal(gen_age_loss_1, self.best_gen_age_loss_1) and
				np.less_equal(gen_age_loss_2, self.best_gen_age_loss_2) and
				np.less_equal(gen_age_loss_3, self.best_gen_age_loss_3) and
				np.less_equal(gen_id_loss_12, self.best_gen_id_loss_12) and
				np.less_equal(gen_id_loss_23, self.best_gen_id_loss_23) and
				np.less_equal(disc_real_loss, self.best_disc_real_loss) and
				np.less_equal(gen_reconn_loss, self.best_gen_reconn_loss) and
				np.less_equal(disc_fake_loss_1, self.best_disc_fake_loss_1) and
				np.less_equal(disc_fake_loss_2, self.best_disc_fake_loss_2) and
				np.less_equal(disc_fake_loss_3, self.best_disc_fake_loss_3) and
				np.less_equal(gen_cycle_loss, self.best_gen_cycle_loss)):
			self.best_gen_id_loss_12 = gen_id_loss_12
			self.best_gen_id_loss_23 = gen_id_loss_23
			self.best_gen_age_loss_1 = gen_age_loss_1
			self.best_gen_age_loss_2 = gen_age_loss_2
			self.best_gen_age_loss_3 = gen_age_loss_3
			self.best_gen_reconn_loss = gen_reconn_loss
			self.best_gen_cycle_loss = gen_cycle_loss
			self.best_disc_real_loss = disc_real_loss
			self.best_disc_fake_loss_1 = disc_fake_loss_1
			self.best_disc_fake_loss_2 = disc_fake_loss_2
			self.best_disc_fake_loss_3 = disc_fake_loss_3
			self.wait = 0

			# Record the best weights if current results is better (less)
			self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		# Prints the stopped epoch at the end of training
		if self.stopped_epoch > 0:
			print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class CustomModelCheckpoint(Callback):
	"""Defines a custom model checkpoint callback that uses multiple conditions"""

	def __init__(self, filepath):
		# Defines the callback variables
		super(CustomModelCheckpoint, self).__init__()
		self.best_gen_id_loss_12 = None
		self.best_gen_id_loss_23 = None
		self.best_gen_age_loss_1 = None
		self.best_gen_age_loss_2 = None
		self.best_gen_age_loss_3 = None
		self.best_gen_cycle_loss = None
		self.best_disc_real_loss = None
		self.best_gen_reconn_loss = None
		self.best_disc_fake_loss_1 = None
		self.best_disc_fake_loss_2 = None
		self.best_disc_fake_loss_3 = None
		self.gen_filepath, self.dis_filepath = filepath

	def on_train_begin(self, logs=None):
		# Initializes the best loss variables to infinity
		self.best_gen_id_loss_12 = np.Inf
		self.best_gen_id_loss_23 = np.Inf
		self.best_gen_age_loss_1 = np.Inf
		self.best_gen_age_loss_2 = np.Inf
		self.best_gen_age_loss_3 = np.Inf
		self.best_gen_reconn_loss = np.Inf
		self.best_gen_cycle_loss = np.Inf
		self.best_disc_real_loss = np.Inf
		self.best_disc_fake_loss_1 = np.Inf
		self.best_disc_fake_loss_2 = np.Inf
		self.best_disc_fake_loss_3 = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		# Checks the loss values to determine whether to save the model
		gen_id_loss_12 = logs.get("val_gen_id_loss_12")
		gen_id_loss_23 = logs.get("val_gen_id_loss_23")
		gen_age_loss_1 = logs.get("val_gen_age_loss_1")
		gen_age_loss_2 = logs.get("val_gen_age_loss_2")
		gen_age_loss_3 = logs.get("val_gen_age_loss_3")
		gen_reconn_loss = logs.get("val_gen_reconn_loss")
		gen_cycle_loss = logs.get("val_gen_cycle_loss")
		disc_real_loss = logs.get("val_disc_real_loss")
		disc_fake_loss_1 = logs.get("val_disc_fake_loss_1")
		disc_fake_loss_2 = logs.get("val_disc_fake_loss_2")
		disc_fake_loss_3 = logs.get("val_disc_fake_loss_3")

		if (np.less_equal(gen_age_loss_1, self.best_gen_age_loss_1) and
				np.less_equal(gen_age_loss_2, self.best_gen_age_loss_2) and
				np.less_equal(gen_age_loss_3, self.best_gen_age_loss_3) and
				np.less_equal(gen_id_loss_12, self.best_gen_id_loss_12) and
				np.less_equal(gen_id_loss_23, self.best_gen_id_loss_23) and
				np.less_equal(disc_real_loss, self.best_disc_real_loss) and
				np.less_equal(gen_reconn_loss, self.best_gen_reconn_loss) and
				np.less_equal(disc_fake_loss_1, self.best_disc_fake_loss_1) and
				np.less_equal(disc_fake_loss_2, self.best_disc_fake_loss_2) and
				np.less_equal(disc_fake_loss_3, self.best_disc_fake_loss_3) and
				np.less_equal(gen_cycle_loss, self.best_gen_cycle_loss)):
			self.best_gen_id_loss_12 = gen_id_loss_12
			self.best_gen_id_loss_23 = gen_id_loss_23
			self.best_gen_age_loss_1 = gen_age_loss_1
			self.best_gen_age_loss_2 = gen_age_loss_2
			self.best_gen_age_loss_3 = gen_age_loss_3
			self.best_gen_reconn_loss = gen_reconn_loss
			self.best_gen_cycle_loss = gen_cycle_loss
			self.best_disc_real_loss = disc_real_loss
			self.best_disc_fake_loss_1 = disc_fake_loss_1
			self.best_disc_fake_loss_2 = disc_fake_loss_2
			self.best_disc_fake_loss_3 = disc_fake_loss_3

			self.model.gen.save(self.gen_filepath)
			self.model.disc.save(self.dis_filepath)


class GANMonitor(Callback):
	"""Defines a callback class to predict the results for a few images after every epoch to monitor the GAN's results"""
	def __init__(self, filepath, data):
		# Initializes the variables
		self.data = data
		self.filepath = filepath

	def on_epoch_end(self, epoch, logs=None):
		# Run the model over a few images and save the results
		if not os.path.exists(self.filepath + "/epoch_" + str(epoch + 1)):
			os.makedirs(self.filepath + "/epoch_" + str(epoch + 1))
		for i, data in enumerate(self.data):
			img, source, target = data["image"], data["source_age_group"], data["target_age_group"]

			# Run the images through the generator model and convert the output to the range [0, 255] before saving
			_, prediction_1 = self.model.gen([img, target])
			prediction_1 = np.rint(prediction_1.numpy()[0] * 127.5 + 127.5).astype(int)
			prediction_1 = np.clip(prediction_1, 0, 255).astype(np.uint8)

			_, prediction_2 = self.model.gen([prediction_1[None, :, :, :], source])
			prediction_2 = np.rint(prediction_2.numpy()[0] * 127.5 + 127.5).astype(int)
			prediction_2 = np.clip(prediction_2, 0, 255).astype(np.uint8)

			_, prediction_3 = self.model.gen([img, source])
			prediction_3 = np.rint(prediction_3.numpy()[0] * 127.5 + 127.5).astype(int)
			prediction_3 = np.clip(prediction_3, 0, 255).astype(np.uint8)

			image = tf.keras.utils.array_to_img(img[0] * 127.5 + 127.5)
			prediction_1 = tf.keras.utils.array_to_img(prediction_1)
			prediction_2 = tf.keras.utils.array_to_img(prediction_2)
			prediction_3 = tf.keras.utils.array_to_img(prediction_3)
			image.save(
				self.filepath + "epoch_{epoch}/{i}_original_img_{source}.png".format(i=i, source=tf.argmax(source[0],
				                                                                                           axis=0),
				                                                                     epoch=epoch + 1)
			)
			prediction_1.save(
				self.filepath + "epoch_{epoch}/{i}_generated_img_{target}.png".format(i=i, target=tf.argmax(target[0],
				                                                                                       axis=0),
				                                                                      epoch=epoch + 1)
			)
			prediction_2.save(
				self.filepath + "epoch_{epoch}/{i}_cycled_img_{source}.png".format(i=i, source=tf.argmax(source[0],
				                                                                                         axis=0),
				                                                                   epoch=epoch + 1)
			)
			prediction_3.save(
				self.filepath + "epoch_{epoch}/{i}_reconstructed_img_{source}.png".format(i=i,
				                                                                          source=tf.argmax(source[0],
				                                                                                           axis=0),
				                                                                          epoch=epoch + 1)
			)
