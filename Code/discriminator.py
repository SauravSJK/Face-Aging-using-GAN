"""Defines the discriminator model."""

from layer_utils import res_block
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.image import per_image_standardization

# Without Leaky ReLu in block
def discriminator(learning_rate=0.0002, job_dir="..", name=None):
    """Define the discriminator model."""
    image_input = Input(shape=(200, 200, 3), name="image")
    x = Conv2D(filters=64, kernel_size=4, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(image_input)
    x = res_block(x, 128, sampling="down")
    x = res_block(x, 256, sampling="down")
    x = res_block(x, 512, sampling="down")
    x = res_block(x, 512, sampling="down")
    x = res_block(x, 512, sampling="down")
    x = res_block(x, 512, sampling="down")
    x = LeakyReLU()(x)
    x = Conv2D(filters=512, kernel_size=3, kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(x)
    x = LeakyReLU()(x)
    x = Reshape((512,))(x)
    output = Dense(14, activation='softmax', kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(x)
    model = Model(inputs=image_input, outputs=output, name=name)
    """model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse")
    model.summary(expand_nested=True, show_trainable=True)
    plot_model(model,
               to_file=job_dir + 'Discriminator.png',
               show_shapes=True,
               expand_nested=True,
               show_layer_activations=True,
               show_trainable=True)"""
    print(model.summary())
    return model
