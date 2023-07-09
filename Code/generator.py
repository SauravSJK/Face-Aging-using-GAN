import tensorflow as tf
from layer_utils import res_block
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.image import per_image_standardization
from tensorflow.keras.losses import sparse_categorical_crossentropy


def encoder():
    input_image = Input(shape=(200, 200, 3))
    x = Conv2D(filters=64,
               kernel_size=4,
               padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(input_image)
    x = res_block(x, 128,
                  sampling="down",
                  norm_type="instance")
    x = res_block(x, 256,
                  sampling="down",
                  norm_type="instance")
    x = res_block(x, 512,
                  sampling="down",
                  norm_type="instance")
    x = res_block(x, 512,
                  sampling="down",
                  sampling_size=5,
                  norm_type="instance")
    x = res_block(x, 512)
    id_ = res_block(x, 512)
    return input_image, id_

def identity(id_):
    age = Input(shape=(14))
    x = res_block(id_, 512,
                  sampling="down",
                  norm_type="cbn",
                  condition=age)
    f_age_aware = Conv2D(filters=512,
                         kernel_size=2,
                         padding="valid", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(x)
    return age, f_age_aware

# Limit output to [0, 255]
def decoder(id_, f_age_aware):
    x = res_block(id_, 512,
                  norm_type="ada_in",
                  condition=f_age_aware)
    x = res_block(x, 512,
                  norm_type="ada_in",
                  condition=f_age_aware)
    x = res_block(x, 512,
                  sampling="up",
                  sampling_size=5,
                  norm_type="ada_in",
                  condition=f_age_aware)
    x = res_block(x, 256,
                  sampling="up",
                  norm_type="ada_in",
                  condition=f_age_aware)
    x = res_block(x, 128,
                  sampling="up",
                  norm_type="ada_in",
                  condition=f_age_aware)
    x = res_block(x, 64,
                  sampling="up",
                  norm_type="ada_in",
                  condition=f_age_aware)
    output_image = Conv2D(3, 1,
                          padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2', activation='sigmoid')(x)
    return output_image

def generator(name=None):
    input_image, id_ = encoder()
    age, f_age_aware = identity(id_)
    output_image = decoder(id_, f_age_aware)
    generator = Model(inputs=[input_image, age], outputs=[id_, output_image], name=name)
    print(generator.summary())
    return generator
"""
# Increase learning rate
def define_gan(gen, dis, learning_rate=0.0002, job_dir=".."):
    dis.trainable = False

    for layer in dis.layers:
        layer._name = layer.name + str("_dis")
    dis._name = "discriminator"

    input_image = Input(shape=(200, 200, 3), name="image")
    source_age = Input(shape=(14), name="source_age_group")
    target_age = Input(shape=(14), name="target_age_group")
    identity_1, output_image_1 = gen([input_image, target_age])
    identity_2, output_image_2 = gen([output_image_1, source_age])
    age_prediction = dis(output_image_1)

    gan = Model(
        inputs=[input_image, source_age, target_age],
        outputs=[identity_1, output_image_1, identity_2,
                 output_image_2, age_prediction])

    identity_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                identity_1,
                identity_2)))
    age_prediction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                target_age,
                age_prediction)))
    reconstruction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                Flatten()(input_image),
                Flatten()(output_image_2))))

    gan.add_loss(identity_loss)
    gan.add_loss(age_prediction_loss)
    gan.add_loss(reconstruction_loss)

    gan.add_metric(
        identity_loss,
        name="identity_loss",
        aggregation="mean")
    gan.add_metric(
        age_prediction_loss,
        name="age_prediction_loss",
        aggregation="mean")
    gan.add_metric(
        reconstruction_loss,
        name="reconstruction_loss",
        aggregation="mean")

    gan.compile(optimizer=Adam(learning_rate=learning_rate))

    #gan.summary(expand_nested=True, show_trainable=True)
    plot_model(gan,
               to_file=job_dir + 'GAN.png',
               show_shapes=True,
               expand_nested=True,
               show_layer_activations=True,
               show_trainable=True)
    return gan"""
