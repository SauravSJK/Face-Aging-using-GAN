import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import get_data
import generator
import numpy as np
import discriminator
from glob import glob
from PIL import Image
import write_tfrecords
import tensorflow as tf

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""

from os.path import exists
from gan import FaceAgeGan
from tensorflow.math import divide
from layer_utils import GANMonitor
from tensorflow.data import AUTOTUNE
from tensorflow.io import decode_jpeg
from tensorflow.io import FixedLenFeature
from layer_utils import CustomEarlyStopping
from tensorflow.data import TFRecordDataset
from tensorflow.keras.layers import Flatten
from layer_utils import CustomModelCheckpoint
from tensorflow.io import parse_single_example
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import BackupAndRestore
from tensorflow.keras.losses import categorical_crossentropy


# Check if loss calc is correct or use something else
def gen_id_loss_fn(identity_1, identity_2):
    identity_loss = tf.reduce_mean(
        tf.sqrt(
            mean_squared_error(
                identity_1,
                identity_2)))
    return identity_loss


def gen_age_loss_fn(disc_fake, target_age):
    """
    age_prediction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                target_age,
                disc_fake)))
    """
    age_prediction_loss = tf.reduce_mean(categorical_crossentropy(target_age, disc_fake))
    return age_prediction_loss


def gen_reconn_loss_fn(input_image, output_image_2):
    reconstruction_loss = tf.reduce_mean(
        tf.sqrt(
            mean_squared_error(
                Flatten()(input_image),
                Flatten()(output_image_2))))
    return reconstruction_loss


"""
def disc_real_age_loss_fn(disc_real, source_age):
    real_age_prediction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                source_age,
                disc_real)))
    return real_age_prediction_loss


def disc_fake_age_loss_fn(disc_fake):
    fake_age_prediction_loss = tf.reduce_sum(
        tf.sqrt(
            mean_squared_error(
                tf.zeros_like(disc_fake),
                disc_fake)))
    return fake_age_prediction_loss
"""

def disc_age_loss_fn(disc_out, target):
    age_prediction_loss = tf.reduce_mean(
        tf.sqrt(
            mean_squared_error(
                target,
                disc_out)))
    return age_prediction_loss

"""
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Get the generators
    gen = generator.generator(name="generator")

    # Get the discriminators
    disc = discriminator.discriminator(name="discriminator")

    # Create cycle gan model
    face_age_gan_model = FaceAgeGan(
        generator=gen,
        discriminator=disc,
        n_gradients=9)

    # Compile the model
    face_age_gan_model.compile(
        gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        gen_id_loss=gen_id_loss_fn,
        gen_age_loss=gen_age_loss_fn,
        gen_reconn_loss=gen_reconn_loss_fn,
        disc_real_age_loss=disc_real_age_loss_fn,
        disc_fake_age_loss=disc_fake_age_loss_fn
    )
    
BATCH_SIZE_PER_REPLICA = 32
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

"""
# Get the generators
gen = generator.generator(name="generator")

# Get the discriminators
disc = discriminator.discriminator(name="discriminator")

# Create cycle gan model
face_age_gan_model = FaceAgeGan(
    generator=gen,
    discriminator=disc,
    n_gradients=9)

# Compile the model
face_age_gan_model.compile(
    gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    gen_id_loss=gen_id_loss_fn,
    gen_age_loss=gen_age_loss_fn,
    gen_reconn_loss=gen_reconn_loss_fn,
    disc_age_loss=disc_age_loss_fn
)

BATCH_SIZE = 32


def parse_tfrecord(example):
    feature_description = {
        "image": FixedLenFeature([], tf.string),
        "source_age_group": FixedLenFeature([], tf.int64),
        "target_age_group": FixedLenFeature([], tf.int64)
    }
    example = parse_single_example(example, feature_description)
    example["image"] = divide(decode_jpeg(example["image"], channels=3), 255)
    example["source_age_group"] = tf.one_hot(example["source_age_group"], 14)
    example["target_age_group"] = tf.one_hot(example["target_age_group"], 14)
    return example


def load_dataset(run_type, job_dir):
    files = sorted(glob(job_dir + "/tfrecords/*"))
    if run_type == "training":
        raw_dataset = TFRecordDataset(files[:-1])
    else:
        raw_dataset = TFRecordDataset(files[-1])
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset


def get_dataset(run_type, job_dir=".."):
    dataset = load_dataset(run_type, job_dir).take(4)
    dataset = dataset.shuffle(2048)
    #dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


tfrecords_dir = ".." + "/tfrecords"
if not exists(tfrecords_dir):
    data = get_data.get_data()
    write_tfrecords.write_tfrecords(data)

plotter_dir = ".." + "/output/"
plotter = GANMonitor(plotter_dir, get_dataset("testing").take(4))

# Setup a callback for Tensorboard to store logs
log_dir = ".." + "/logs/gan/"
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1)

# Setup a callback for Model Checkpointing
gen_checkpoint_dir = ".." + "/checkpoint/generator/"
dis_checkpoint_dir = ".." + "/checkpoint/discriminator/"
model_checkpoint_callback = CustomModelCheckpoint(
    filepath=[gen_checkpoint_dir, dis_checkpoint_dir])

# Setup a callback for Backup and Restore
backup_dir = ".." + "/backup/"
backup_restore_callback = BackupAndRestore(
    backup_dir,
    save_freq="epoch",
    delete_checkpoint=False,
    save_before_preemption=False)

early_stopping_callback = CustomEarlyStopping(
    patience=20)

face_age_gan_model.fit(
    get_dataset("training"),
    epochs=100,
    verbose=2,
    callbacks=[
        plotter,
        tensorboard_callback,
        model_checkpoint_callback,
        backup_restore_callback,
        early_stopping_callback],
    validation_data=get_dataset("testing"))