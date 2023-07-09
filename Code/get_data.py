import numpy as np
import pandas as pd
from hurry.filesize import size
from os import listdir, system, path


def read_dataset(job_dir=".."):
    if not path.exists(job_dir + "/UTKFace"):
        system("gdown --fuzzy \"https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=share_link&resourcekey=0-dabpv_3J0C0cditpiAfhAw\"")
        system("tar -xf UTKFace.tar.gz")
        system("rm UTKFace.tar.gz")
        system("mv UTKFace " + job_dir + "/")
    age = []
    gender = []
    race = []
    img_path = []
    sizes = []
    for file in listdir(job_dir + "/UTKFace"):
        temp_size = size(path.getsize(job_dir + "/UTKFace/" + file))
        if temp_size not in ['2K', '3K', '4K']:
            name = file.split('_')
            # Some files do not have race in the name, so skip them (just 3 files)
            if len(name) != 4:
                continue
            age.append(int(name[0]))
            gender.append(int(name[1]))
            race.append(int(name[2]))
            img_path.append(file)
            sizes.append(temp_size)
    return pd.DataFrame({
        'age': age,
        'gender': gender,
        'race': race,
        'img': img_path, 
        'size': sizes})


def group_age(data):
    conditions = [
        (data['age'] <= 5),
        (data['age'] > 5) & (data['age'] <= 10),
        (data['age'] > 10) & (data['age'] <= 15),
        (data['age'] > 15) & (data['age'] <= 20),
        (data['age'] > 20) & (data['age'] <= 25),
        (data['age'] > 25) & (data['age'] <= 30),
        (data['age'] > 30) & (data['age'] <= 40),
        (data['age'] > 40) & (data['age'] <= 50),
        (data['age'] > 50) & (data['age'] <= 60),
        (data['age'] > 60) & (data['age'] <= 70),
        (data['age'] > 70) & (data['age'] <= 80),
        (data['age'] > 80) & (data['age'] <= 90),
        (data['age'] > 90) & (data['age'] <= 100),
        (data['age'] > 100)]

    # create a list of the values we want to assign for each condition
    age_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # create a new column and use np.select to assign values to it using our lists as arguments
    return np.select(conditions, age_groups)


def get_data(job_dir=".."):
    data = read_dataset(job_dir)
    data["age_group"] = group_age(data)
    return data
