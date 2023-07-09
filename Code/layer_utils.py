import numpy as np
import tensorflow as tf
from os import makedirs
from os.path import exists
from tensorflow.nn import moments
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.nn import batch_normalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import Ones
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import AveragePooling2D
#from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import GroupNormalization


# Ref: https://colab.research.google.com/drive/1WGG8d22KoxXWBThYOeFDcHvt_z9EirHV#scrollTo=-CxyRhZaDSYk
class ConditionBatchNormalization(Layer):
    def __init__(self):
        super(ConditionBatchNormalization, self).__init__()
        self.decay = 0.9
        self.epsilon = 1e-05

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.beta_mapping = Dense(self.num_channels, kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')
        self.gamma_mapping = Dense(self.num_channels, kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')
        self.test_mean = tf.Variable(initial_value=Zeros()(self.num_channels), trainable=False, dtype=tf.float32)
        self.test_var = tf.Variable(initial_value=Ones()(self.num_channels), trainable=False, dtype=tf.float32)

    def call(self, x, training=None):
        #Generate beta, gamma
        x, conditions = x
        beta = self.beta_mapping(conditions)
        gamma = self.gamma_mapping(conditions)

        beta = tf.reshape(beta, shape=[-1, 1, 1, self.num_channels])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.num_channels])
        if training:
            #Calculate mean and varience of X.
            batch_mean, batch_var = moments(x, [0, 1, 2])
            #Calculate parameters for test set
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
        config = super().get_config()
        config.update({"units": self.num_channels})
        return config


# Ref: https://github.com/ariG23498/AdaIN-TF/blob/master/AdaIN.ipynb
def ada_in(style, content, epsilon=1e-5):
    axes = [1, 2]

    c_mean, c_var = moments(content, axes=axes, keepdims=True)
    s_mean, s_var = moments(style, axes=axes, keepdims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    t = s_std * (content - c_mean) / c_std + s_mean
    return t


def res_block(x, filters, kernelsize=4, sampling=None, sampling_size=2, norm_type=None, condition=None):
    if norm_type == "instance":
        #y = InstanceNormalization()(x)
        y = GroupNormalization(groups=-1)(x)
    elif norm_type == "cbn":
        y = ConditionBatchNormalization()([x, condition])
    elif norm_type == "ada_in":
        condition = Conv2D(x.shape[3], 1, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(condition)
        y = ada_in(condition, x)
    else:
        y = x
    y = LeakyReLU()(y)
    y = Conv2D(filters, kernelsize, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(y)
    if norm_type == "instance":
        #y = InstanceNormalization()(y)
        y = GroupNormalization(groups=-1)(x)
    elif norm_type == "cbn":
        y = ConditionBatchNormalization()([x, condition])
    elif norm_type == "ada_in":
        y = ada_in(condition, x)
    else:
        y = x
    y = LeakyReLU()(y)
    y = Conv2D(filters, kernelsize, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(y)
    z = Conv2D(filters, 1, padding="same", kernel_regularizer='l1_l2', bias_regularizer='l1_l2', activity_regularizer='l1_l2')(x)
    out = y + z
    if sampling == "down":
        return AveragePooling2D(pool_size=sampling_size)(out)
    elif sampling == "up":
        return UpSampling2D(size=sampling_size)(out)
    else:
        return out


# Ref: https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
class CustomEarlyStopping(Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_gen_loss = np.Inf
        self.best_gen_id_loss = np.Inf
        self.best_gen_age_loss = np.Inf
        self.best_gen_reconn_loss = np.Inf
        self.best_disc_loss = np.Inf
        """
        self.best_disc_real_age_loss = np.Inf
        self.best_disc_fake_age_loss = np.Inf
        """

    def on_epoch_end(self, epoch, logs=None):
        gen_loss = logs.get("val_gen_loss")
        gen_id_loss = logs.get("val_gen_id_loss")
        gen_age_loss = logs.get("val_gen_age_loss")
        gen_reconn_loss = logs.get("val_gen_reconn_loss")
        disc_loss = logs.get("val_disc_loss")
        """
        disc_real_age_loss = logs.get("val_disc_real_age_loss")
        disc_fake_age_loss = logs.get("val_disc_fake_age_loss")
        """

        # If BOTH the validation loss AND map10 does not improve for 'patience' epochs, stop training early.
        """
        if np.less(gen_id_loss, self.best_gen_id_loss) and \
            np.less(gen_age_loss, self.best_gen_age_loss) and \
                np.less(gen_reconn_loss, self.best_gen_reconn_loss) and \
                    np.less(disc_real_age_loss, self.best_disc_real_age_loss) and \
                        np.less(disc_fake_age_loss, self.best_disc_fake_age_loss):
            self.best_gen_id_loss = gen_id_loss
            self.best_gen_age_loss = gen_age_loss
            self.best_gen_reconn_loss = gen_reconn_loss
            self.best_disc_real_age_loss = disc_real_age_loss
            self.best_disc_fake_age_loss = disc_fake_age_loss
        """
        if np.less(gen_loss, self.best_gen_loss) and \
            np.less(disc_loss, self.best_disc_loss):
            self.best_gen_loss = gen_loss
            self.best_disc_loss = disc_loss
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            
class CustomModelCheckpoint(Callback):
    def __init__(self, filepath):
        super(CustomModelCheckpoint, self).__init__()
        self.gen_filepath, self.dis_filepath = filepath
    
    def on_train_begin(self, logs=None):
        self.best_gen_loss = np.Inf
        self.best_gen_id_loss = np.Inf
        self.best_gen_age_loss = np.Inf
        self.best_gen_reconn_loss = np.Inf
        self.best_disc_loss = np.Inf
        """
        self.best_disc_real_age_loss = np.Inf
        self.best_disc_fake_age_loss = np.Inf
        """
    
    def on_epoch_end(self, epoch, logs=None):
        gen_loss = logs.get("val_gen_loss")
        gen_id_loss = logs.get("val_gen_id_loss")
        gen_age_loss = logs.get("val_gen_age_loss")
        gen_reconn_loss = logs.get("val_gen_reconn_loss")
        disc_loss = logs.get("val_disc_loss")
        """
        disc_real_age_loss = logs.get("val_disc_real_age_loss")
        disc_fake_age_loss = logs.get("val_disc_fake_age_loss")
        """

        # If BOTH the validation loss AND map10 does not improve for 'patience' epochs, stop training early.
        """
        if np.less(gen_id_loss, self.best_gen_id_loss) and \
            np.less(gen_age_loss, self.best_gen_age_loss) and \
                np.less(gen_reconn_loss, self.best_gen_reconn_loss) and \
                    np.less(disc_real_age_loss, self.best_disc_real_age_loss) and \
                        np.less(disc_fake_age_loss, self.best_disc_fake_age_loss):
            self.best_gen_id_loss = gen_id_loss
            self.best_gen_age_loss = gen_age_loss
            self.best_gen_reconn_loss = gen_reconn_loss
            self.best_disc_real_age_loss = disc_real_age_loss
            self.best_disc_fake_age_loss = disc_fake_age_loss
        """
        if np.less(gen_loss, self.best_gen_loss) and \
            np.less(disc_loss, self.best_disc_loss):
            self.best_gen_loss = gen_loss
            self.best_disc_loss = disc_loss
            
            self.model.gen.save(self.gen_filepath)
            self.model.disc.save(self.dis_filepath)
            
class GANMonitor(Callback):
    def __init__(self, filepath, data):
        self.filepath = filepath
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        if not exists(self.filepath + "/epoch_" + str(epoch + 1)):
            makedirs(self.filepath + "/epoch_" + str(epoch + 1))
        rand_img = tf.random.uniform([200, 200, 3], maxval=1.2)
        rand_prediction = np.rint(rand_img * 255).astype(int)
        rand_prediction = np.clip(rand_prediction, 0, 255).astype(np.uint8)

        rand_prediction = tf.keras.utils.array_to_img(rand_prediction)
        rand_prediction.save(
            self.filepath + "epoch_{epoch}/test_image.png".format(epoch=epoch + 1)
        )
        for i, data in enumerate(self.data):
            img, source, target = data["image"], data["source_age_group"], data["target_age_group"]
            _, prediction = self.model.gen([img, target])
            prediction = np.rint(prediction.numpy()[0] * 255).astype(int)
            prediction = np.clip(prediction, 0, 255).astype(np.uint8)

            image = tf.keras.utils.array_to_img(img[0])
            prediction = tf.keras.utils.array_to_img(prediction)
            image.save(
                self.filepath + "epoch_{epoch}/{i}_original_img_{source}.png".format(i=i, source=tf.argmax(source[0], axis=0), epoch=epoch + 1)
            )
            prediction.save(
                self.filepath + "epoch_{epoch}/{i}_generated_img_{target}.png".format(i=i, target=tf.argmax(target[0], axis=0), epoch=epoch + 1)
            )
