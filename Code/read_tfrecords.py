"""Defines the functions used for reading the tfrecords"""

import os
import get_data
from glob import glob
import write_tfrecords
import tensorflow as tf

def parse_tfrecord(example):
	# Defines the function for parsing the tfrecord
	feature_description = {"image": tf.io.FixedLenFeature([], tf.string),
	                       "source_age_group": tf.io.FixedLenFeature([], tf.int64),
	                       "target_age_group": tf.io.FixedLenFeature([], tf.int64)
	                       }
	example = tf.io.parse_single_example(example, feature_description)
	example["image"] = 2 * tf.math.divide(tf.io.decode_jpeg(example["image"], channels=3), 255) - 1
	example["source_age_group"] = tf.one_hot(example["source_age_group"], 14)
	example["target_age_group"] = tf.one_hot(example["target_age_group"], 14)
	return example


def load_dataset(run_type, job_dir=".."):
	# Defines the files to be read as part of the training and testing sets and returns the parsed data
	files = sorted(glob(job_dir + "/tfrecords/*"))
	if run_type == "training":
		raw_dataset = tf.data.TFRecordDataset(files[:-1])
	else:
		raw_dataset = tf.data.TFRecordDataset(files[-1])
	parsed_dataset = raw_dataset.map(parse_tfrecord)
	return parsed_dataset


def process_dataset(batch_size, run_type, job_dir=".."):
	# Defines the dataset processing functions to be run on the data
	dataset = load_dataset(run_type, job_dir)
	dataset = dataset.shuffle(2048)
	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
	return dataset


def get_dataset(batch_size, job_dir=".."):
	# Generates the tfrecords if not already available
	tfrecords_dir = job_dir + "/tfrecords"
	if not os.path.exists(tfrecords_dir):
		data = get_data.get_data(job_dir)
		write_tfrecords.write_tfrecords(data, job_dir)
	return process_dataset(batch_size, "training", job_dir), process_dataset(batch_size, "testing", job_dir)