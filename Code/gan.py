"""Defines the GAN model"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean


class FaceAgeGan(Model):
	"""GAN Model to combine the generator and discriminator"""
	def __init__(self, generator, discriminator):
		# Initializes the model variables: sub-models, optimizers, loss functions, and metrics
		super().__init__()
		self.gen = generator
		self.disc = discriminator
		self.gen_optimizer = None
		self.gen_id_loss_fn = None
		self.disc_optimizer = None
		self.real_age_loss_fn = None
		self.fake_age_loss_fn = None
		self.gen_img_loss_fn = None
		self.g_id_loss_12_test_tracker = Mean(name="gen_id_test_loss_12")
		self.g_id_loss_12_train_tracker = Mean(name="gen_id_train_loss_12")
		self.g_id_loss_23_test_tracker = Mean(name="gen_id_test_loss_23")
		self.g_id_loss_23_train_tracker = Mean(name="gen_id_train_loss_23")
		self.g_age_loss_1_test_tracker = Mean(name="gen_age_test_loss_1")
		self.g_age_loss_1_train_tracker = Mean(name="gen_age_train_loss_1")
		self.g_age_loss_2_test_tracker = Mean(name="gen_age_test_loss_2")
		self.g_age_loss_2_train_tracker = Mean(name="gen_age_train_loss_2")
		self.g_age_loss_3_test_tracker = Mean(name="gen_age_test_loss_3")
		self.g_age_loss_3_train_tracker = Mean(name="gen_age_train_loss_3")
		self.d_real_loss_test_tracker = Mean(name="disc_real_test_loss")
		self.d_real_loss_train_tracker = Mean(name="disc_real_train_loss")
		self.d_fake_loss_1_test_tracker = Mean(name="disc_fake_test_loss_1")
		self.d_fake_loss_1_train_tracker = Mean(name="disc_fake_train_loss_1")
		self.d_fake_loss_2_test_tracker = Mean(name="disc_fake_test_loss_2")
		self.d_fake_loss_2_train_tracker = Mean(name="disc_fake_train_loss_2")
		self.d_fake_loss_3_test_tracker = Mean(name="disc_fake_test_loss_3")
		self.d_fake_loss_3_train_tracker = Mean(name="disc_fake_train_loss_3")
		self.g_cycle_loss_test_tracker = Mean(name="gen_cycle_test_loss")
		self.g_cycle_loss_train_tracker = Mean(name="gen_cycle_train_loss")
		self.g_reconn_loss_test_tracker = Mean(name="gen_reconn_test_loss")
		self.g_reconn_loss_train_tracker = Mean(name="gen_reconn_train_loss")

	def compile(self, gen_optimizer, disc_optimizer, gen_id_loss, real_age_loss, gen_img_loss, fake_age_loss):
		# Assigns the optimizers and loss functions to the model variables
		super().compile()
		self.gen_optimizer = gen_optimizer
		self.disc_optimizer = disc_optimizer
		self.gen_id_loss_fn = gen_id_loss
		self.real_age_loss_fn = real_age_loss
		self.gen_img_loss_fn = gen_img_loss
		self.fake_age_loss_fn = fake_age_loss

	def train_step(self, data):
		# Defines a single step of the training loop
		# Initialize the input data variables
		input_image, source_age, target_age = data["image"], data["source_age_group"], data["target_age_group"]

		# Phase 1: Train the discriminator
		# Run the input data through the generator model
		_, output_image_1 = self.gen([input_image, target_age], training=False)
		_, output_image_2 = self.gen([output_image_1, source_age], training=False)
		_, output_image_3 = self.gen([input_image, source_age], training=False)
		with tf.GradientTape() as tape:
			# Run the generator's output through the discriminator
			disc_real = self.disc(input_image, training=True)
			disc_fake_1 = self.disc(output_image_1, training=True)
			disc_fake_2 = self.disc(output_image_2, training=True)
			disc_fake_3 = self.disc(output_image_3, training=True)

			# Calculate the loss using the assigned loss functions
			disc_fake_loss_1 = self.fake_age_loss_fn(disc_fake_1, tf.zeros_like(disc_fake_1) + 0.05 * tf.random.uniform(
				tf.shape(disc_fake_1)))
			disc_fake_loss_2 = self.fake_age_loss_fn(disc_fake_2, tf.zeros_like(disc_fake_2) + 0.05 * tf.random.uniform(
				tf.shape(disc_fake_2)))
			disc_fake_loss_3 = self.fake_age_loss_fn(disc_fake_3, tf.zeros_like(disc_fake_3) + 0.05 * tf.random.uniform(
				tf.shape(disc_fake_3)))
			disc_real_loss = self.real_age_loss_fn(disc_real, source_age)
			disc_loss = (disc_fake_loss_1 + disc_fake_loss_2 + disc_fake_loss_3 + 3 * disc_real_loss) / 6

		# Calculate the gradients and updates the weights
		disc_grads = tape.gradient(disc_loss, self.disc.trainable_weights)
		self.disc_optimizer.apply_gradients(
			zip(disc_grads, self.disc.trainable_weights)
		)

		# Phase 2: Train the generator
		with tf.GradientTape() as tape:
			# Get the generator's output
			identity_1, output_image_1 = self.gen([input_image, target_age], training=True)
			identity_2, output_image_2 = self.gen([output_image_1, source_age], training=True)
			identity_3, output_image_3 = self.gen([input_image, source_age], training=True)
			# Run the generator's output through the discriminator
			disc_fake_1 = self.disc(output_image_1, training=False)
			disc_fake_2 = self.disc(output_image_2, training=False)
			disc_fake_3 = self.disc(output_image_3, training=False)

			# Calculate the different losses
			gen_age_loss_1 = self.real_age_loss_fn(disc_fake_1, target_age)
			gen_age_loss_2 = self.real_age_loss_fn(disc_fake_2, source_age)
			gen_age_loss_3 = self.real_age_loss_fn(disc_fake_3, source_age)
			gen_id_loss_12 = self.gen_id_loss_fn(identity_1, identity_2)
			gen_id_loss_23 = self.gen_id_loss_fn(identity_2, identity_3)
			gen_cycle_loss = self.gen_img_loss_fn(input_image, output_image_2)
			gen_reconn_loss = self.gen_img_loss_fn(input_image, output_image_3)
			gen_loss = gen_age_loss_1 + gen_age_loss_2 + gen_age_loss_3 + gen_id_loss_12 + gen_id_loss_23 + gen_reconn_loss + gen_cycle_loss

		# Calculate the gradients and update the weights
		gen_grads = tape.gradient(gen_loss, self.gen.trainable_weights)
		self.gen_optimizer.apply_gradients(
			zip(gen_grads, self.gen.trainable_weights)
		)

		# Update the epoch-wise training metrics using the calculated batch-wise loss
		self.d_real_loss_train_tracker.update_state(disc_real_loss)
		self.d_fake_loss_1_train_tracker.update_state(disc_fake_loss_1)
		self.d_fake_loss_2_train_tracker.update_state(disc_fake_loss_2)
		self.d_fake_loss_3_train_tracker.update_state(disc_fake_loss_3)
		self.g_age_loss_1_train_tracker.update_state(gen_age_loss_1)
		self.g_age_loss_2_train_tracker.update_state(gen_age_loss_2)
		self.g_age_loss_3_train_tracker.update_state(gen_age_loss_3)
		self.g_id_loss_12_train_tracker.update_state(gen_id_loss_12)
		self.g_id_loss_23_train_tracker.update_state(gen_id_loss_23)
		self.g_reconn_loss_train_tracker.update_state(gen_reconn_loss)
		self.g_cycle_loss_train_tracker.update_state(gen_cycle_loss)

		return {
			"gen_age_loss_1": self.g_age_loss_1_train_tracker.result(),
			"gen_age_loss_2": self.g_age_loss_2_train_tracker.result(),
			"gen_age_loss_3": self.g_age_loss_3_train_tracker.result(),
			"gen_id_loss_12": self.g_id_loss_12_train_tracker.result(),
			"gen_id_loss_23": self.g_id_loss_23_train_tracker.result(),
			"gen_reconn_loss": self.g_reconn_loss_train_tracker.result(),
			"gen_cycle_loss": self.g_cycle_loss_train_tracker.result(),
			"disc_real_loss": self.d_real_loss_train_tracker.result(),
			"disc_fake_loss_1": self.d_fake_loss_1_train_tracker.result(),
			"disc_fake_loss_2": self.d_fake_loss_2_train_tracker.result(),
			"disc_fake_loss_3": self.d_fake_loss_3_train_tracker.result()
		}

	def test_step(self, data):
		# Defines a single step of the testing loop
		# Get the input data
		input_image, source_age, target_age = data["image"], data["source_age_group"], data["target_age_group"]

		# Run the input through the generator
		identity_1, output_image_1 = self.gen([input_image, target_age], training=False)
		identity_2, output_image_2 = self.gen([output_image_1, source_age], training=False)
		identity_3, output_image_3 = self.gen([input_image, source_age], training=False)

		# Get the discriminator's output using the generator's output
		disc_real = self.disc(input_image, training=False)
		disc_fake_1 = self.disc(output_image_1, training=False)
		disc_fake_2 = self.disc(output_image_2, training=False)
		disc_fake_3 = self.disc(output_image_3, training=False)

		# Calculate the generator's loss
		gen_id_loss_12 = self.gen_id_loss_fn(identity_1, identity_2)
		gen_id_loss_23 = self.gen_id_loss_fn(identity_2, identity_3)
		gen_age_loss_1 = self.real_age_loss_fn(disc_fake_1, target_age)
		gen_age_loss_2 = self.real_age_loss_fn(disc_fake_2, source_age)
		gen_age_loss_3 = self.real_age_loss_fn(disc_fake_3, source_age)
		gen_cycle_loss = self.gen_img_loss_fn(input_image, output_image_2)
		gen_reconn_loss = self.gen_img_loss_fn(input_image, output_image_3)

		# Calculate the discriminator's loss
		disc_fake_loss_1 = self.fake_age_loss_fn(disc_fake_1, tf.zeros_like(disc_fake_1))
		disc_fake_loss_2 = self.fake_age_loss_fn(disc_fake_2, tf.zeros_like(disc_fake_2))
		disc_fake_loss_3 = self.fake_age_loss_fn(disc_fake_3, tf.zeros_like(disc_fake_3))
		disc_real_loss = self.real_age_loss_fn(disc_real, source_age)

		# Update the epoch-wise testing metrics using the batch-wise loss
		self.d_real_loss_test_tracker.update_state(disc_real_loss)
		self.d_fake_loss_1_test_tracker.update_state(disc_fake_loss_1)
		self.d_fake_loss_2_test_tracker.update_state(disc_fake_loss_2)
		self.d_fake_loss_3_test_tracker.update_state(disc_fake_loss_3)
		self.g_age_loss_1_test_tracker.update_state(gen_age_loss_1)
		self.g_age_loss_2_test_tracker.update_state(gen_age_loss_2)
		self.g_age_loss_3_test_tracker.update_state(gen_age_loss_3)
		self.g_id_loss_12_test_tracker.update_state(gen_id_loss_12)
		self.g_id_loss_23_test_tracker.update_state(gen_id_loss_23)
		self.g_reconn_loss_test_tracker.update_state(gen_reconn_loss)
		self.g_cycle_loss_test_tracker.update_state(gen_cycle_loss)

		return {
			"gen_age_loss_1": self.g_age_loss_1_test_tracker.result(),
			"gen_age_loss_2": self.g_age_loss_2_test_tracker.result(),
			"gen_age_loss_3": self.g_age_loss_3_test_tracker.result(),
			"gen_id_loss_12": self.g_id_loss_12_test_tracker.result(),
			"gen_id_loss_23": self.g_id_loss_23_test_tracker.result(),
			"gen_reconn_loss": self.g_reconn_loss_test_tracker.result(),
			"gen_cycle_loss": self.g_cycle_loss_test_tracker.result(),
			"disc_real_loss": self.d_real_loss_test_tracker.result(),
			"disc_fake_loss_1": self.d_fake_loss_1_test_tracker.result(),
			"disc_fake_loss_2": self.d_fake_loss_2_test_tracker.result(),
			"disc_fake_loss_3": self.d_fake_loss_3_test_tracker.result()
		}
