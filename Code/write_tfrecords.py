import os
from os.path import exists
from tensorflow.io import read_file
from tensorflow.train import Feature
from tensorflow.train import Example
from tensorflow.train import Features
from tensorflow.io import encode_jpeg
from tensorflow.io import decode_jpeg
from tensorflow.train import Int64List
from tensorflow.train import BytesList
from tensorflow.io import TFRecordWriter

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return Feature(
        bytes_list=BytesList(value=[encode_jpeg(value).numpy()])
    )

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return Feature(int64_list=Int64List(value=[value]))

def create_example(image, source_age_group, target_age_group):
    feature = {
        "image": image_feature(image),
        "source_age_group": int64_feature(source_age_group),
        "target_age_group": int64_feature(target_age_group)
    }
    return Example(features=Features(feature=feature))


def tf_writer(data, num_tfrecords, num_samples, tfrecords_dir, folder_path, age_groups):
    for tfrec_num in range(num_tfrecords):
        samples = data.iloc[
            (tfrec_num * num_samples): ((tfrec_num + 1) * num_samples)]

        with TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for index, sample in samples.iterrows():
                image_path = folder_path + "/" + sample["img"]
                image = decode_jpeg(read_file(image_path))
                for value in age_groups:
                    if sample["age_group"] != value:
                        example = create_example(image,
                                                 sample["age_group"],
                                                 value)
                        writer.write(example.SerializeToString())


def write_tfrecords(data, job_dir=".."):
    num_samples = 4096
    num_tfrecords = data.shape[0] // num_samples
    tfrecords_dir = job_dir + "/tfrecords"
    folder_path = job_dir + "/UTKFace"
    age_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if data.shape[0] % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples

    if not exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder

    tf_writer(
        data.sample(frac=1, random_state=12),
        num_tfrecords,
        num_samples,
        tfrecords_dir,
        folder_path,
        age_groups)
