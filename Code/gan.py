import tensorflow as tf
from tensorflow.keras import Model


class FaceAgeGan(Model):
    def __init__(
        self,
        generator,
        discriminator,
        n_gradients=8
    ):
        super().__init__()
        self.gen = generator
        self.disc = discriminator
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gen_gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.gen.trainable_variables]
        self.disc_gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.disc.trainable_variables]

    def compile(
        self,
        gen_optimizer,
        disc_optimizer,
        gen_id_loss,
        gen_age_loss,
        gen_reconn_loss,
        disc_age_loss
    ):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_id_loss_fn = gen_id_loss
        self.gen_age_loss_fn = gen_age_loss
        self.gen_reconn_loss_fn = gen_reconn_loss
        self.disc_age_loss_fn = disc_age_loss
        """
        self.disc_real_age_loss_fn = disc_real_age_loss
        self.disc_fake_age_loss_fn = disc_fake_age_loss
        """

    #removed for distributed training
    #@tf.function
    def train_step(self, data):
        self.n_acum_step.assign_add(1)
        # x is Horse and y is zebra
        input_image, source_age, target_age = data["image"], data["source_age_group"], data["target_age_group"]

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            identity_1, output_image_1 = self.gen([input_image, target_age], training=True)
            identity_2, output_image_2 = self.gen([output_image_1, source_age], training=True)

            # Discriminator output
            disc_real = self.disc(input_image, training=True)
            disc_fake = self.disc(output_image_1, training=True)

            gen_id_loss = self.gen_id_loss_fn(identity_1, identity_2)
            gen_age_loss = self.gen_age_loss_fn(disc_fake, target_age)
            gen_reconn_loss = self.gen_reconn_loss_fn(input_image, output_image_2)
            gen_loss = gen_id_loss + gen_age_loss + gen_reconn_loss

            """
            disc_real_age_loss = self.disc_real_age_loss_fn(disc_real, source_age)
            disc_fake_age_loss = self.disc_fake_age_loss_fn(disc_fake)
            disc_loss = disc_real_age_loss + disc_fake_age_loss
            """

            disc_loss = self.disc_age_loss_fn(tf.concat([disc_real, disc_fake], 0), tf.concat([source_age, tf.zeros_like(disc_fake)], 0))


        # Get the gradients for the generators
        gen_grads = tape.gradient(gen_loss, self.gen.trainable_variables)

        # Get the gradients for the discriminators
        disc_grads = tape.gradient(disc_loss, self.disc.trainable_variables)

        #removed for distributed training
        
        for i in range(len(self.gen_gradient_accumulation)):
            self.gen_gradient_accumulation[i].assign_add(gen_grads[i] / tf.cast(self.n_gradients, tf.float32))

        for i in range(len(self.disc_gradient_accumulation)):
            self.disc_gradient_accumulation[i].assign_add(disc_grads[i] / tf.cast(self.n_gradients, tf.float32))

        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        """
        # Update the weights of the generators
        self.gen_optimizer.apply_gradients( 
            zip(gen_grads, self.gen.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.disc.trainable_variables)
        )
        """

        return {
            "gen_loss": gen_loss,
            "gen_id_loss": gen_id_loss,
            "gen_age_loss": gen_age_loss,
            "gen_reconn_loss": gen_reconn_loss,
            "disc_loss": disc_loss
        }

    def test_step(self, data):
        input_image, source_age, target_age = data["image"], data["source_age_group"], data["target_age_group"]
        identity_1, output_image_1 = self.gen([input_image, target_age], training=False)
        identity_2, output_image_2 = self.gen([output_image_1, source_age], training=False)

        # Discriminator output
        disc_real = self.disc(input_image, training=False)
        disc_fake = self.disc(output_image_1, training=False)

        gen_id_loss = self.gen_id_loss_fn(identity_1, identity_2)
        gen_age_loss = self.gen_age_loss_fn(disc_fake, target_age)
        gen_reconn_loss = self.gen_reconn_loss_fn(input_image, output_image_2)
        gen_loss = gen_id_loss + gen_age_loss + gen_reconn_loss
        
        """
        disc_real_age_loss = self.disc_real_age_loss_fn(disc_real, source_age)
        disc_fake_age_loss = self.disc_fake_age_loss_fn(disc_fake)
        disc_loss = disc_real_age_loss + disc_fake_age_loss
        """

        disc_loss = self.disc_age_loss_fn(tf.concat([disc_real, disc_fake], 0), tf.concat([source_age, tf.zeros_like(disc_fake)], 0))
        

        return {
            "gen_loss": gen_loss,
            "gen_id_loss": gen_id_loss,
            "gen_age_loss": gen_age_loss,
            "gen_reconn_loss": gen_reconn_loss,
            "disc_loss": disc_loss
        }

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.gen_optimizer.apply_gradients(zip(self.gen_gradient_accumulation, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(self.disc_gradient_accumulation, self.disc.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gen_gradient_accumulation)):
            self.gen_gradient_accumulation[i].assign(tf.zeros_like(self.gen.trainable_variables[i], dtype=tf.float32))
            
        for i in range(len(self.disc_gradient_accumulation)):
            self.disc_gradient_accumulation[i].assign(tf.zeros_like(self.disc.trainable_variables[i], dtype=tf.float32))
