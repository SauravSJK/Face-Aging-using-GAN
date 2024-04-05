"""Initializes the GAN model and starts the training"""

import os
# Set log level to 3 to print only minimum logs to console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gan
import generator
import discriminator
import custom_losses
import read_tfrecords
import tensorflow as tf
import custom_callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import BackupAndRestore
from gradient_accumulator import GradientAccumulateOptimizer  # https://github.com/andreped/GradientAccumulator


def set_gpu_memory():
	# Get the number of available GPUs and set the same memory growth on them to stabilize training
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
			return True
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)
			return False


def get_callbacks(test_data, monitor_num_imgs=4, hist_freq=1, patience=1000, job_dir=".."):
	# Defines the callbacks
	# Set up a callback for intermittent saving of model outputs
	gan_monitor_dir = job_dir + "/output/"
	gan_monitor_callback = custom_callbacks.GANMonitor(gan_monitor_dir, test_data.take(monitor_num_imgs))

	# Set up a callback for Tensorboard to store logs
	log_dir = job_dir + "/logs/gan/"
	tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=hist_freq)

	# Set up a callback for Model Checkpointing
	gen_checkpoint_dir = job_dir + "/checkpoint/generator/"
	dis_checkpoint_dir = job_dir + "/checkpoint/discriminator/"
	model_checkpoint_callback = custom_callbacks.CustomModelCheckpoint(
		filepath=[gen_checkpoint_dir, dis_checkpoint_dir])

	# Set up a callback for Backup and Restore
	backup_dir = job_dir + "/backup/"
	backup_restore_callback = BackupAndRestore(backup_dir, save_freq="epoch", delete_checkpoint=False,
	                                           save_before_preemption=False)

	# Set up a callback for Early Stopping
	early_stopping_callback = custom_callbacks.CustomEarlyStopping(patience=patience)

	return [gan_monitor_callback, tensorboard_callback, model_checkpoint_callback, backup_restore_callback,
	        early_stopping_callback]


def train(batch_size_per_replica=32, gen_lr=0.00001, disc_lr=0.00001, epochs=1000, patience=1000, job_dir=".."):
	# Initializes the distributed strategy, compiles the model, and runs the fit function
	if not set_gpu_memory():
		return None
	# Define the model under a distributed strategy instance to utilize multiple GPUs, if available
	strategy = tf.distribute.MirroredStrategy()
	batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
	with strategy.scope():
		# If the available GPU memory space is small, use Gradient Accumulator to simulate a larger batch size
		if batch_size < 256:
			gen_optimizer = GradientAccumulateOptimizer(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=gen_lr),
			                                          accum_steps=256 // batch_size)
			disc_optimizer = GradientAccumulateOptimizer(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=disc_lr),
			                                           accum_steps=256 // batch_size)
		else:
			gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
			disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)

		gen = generator.generator(name="generator")
		disc = discriminator.discriminator(name="discriminator")
		face_age_gan_model = gan.FaceAgeGan(generator=gen, discriminator=disc)
		face_age_gan_model.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
		                           gen_id_loss=custom_losses.gen_id_loss_fn,
		                           real_age_loss=custom_losses.real_age_loss_fn,
		                           gen_img_loss=custom_losses.gen_img_loss_fn,
		                           fake_age_loss=custom_losses.fake_age_loss_fn)


	train_data, test_data = read_tfrecords.get_dataset(batch_size=batch_size, job_dir=job_dir)
	face_age_gan_model.fit(train_data, epochs=epochs, verbose=2, callbacks=get_callbacks(test_data=test_data,
	                                                                                   patience=patience),
	                       validation_data=test_data)