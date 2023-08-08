"""Defines the functions to get the data"""

import os
import numpy as np
import pandas as pd
from hurry.filesize import size


def read_dataset(job_dir: str = "..") -> pd.DataFrame:
    # Downloads the dataset and reads them into a dataframe
    if not os.path.exists(job_dir + "/UTKFace"):
        os.system("gdown --fuzzy \"https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=share_link&resourcekey=0-dabpv_3J0C0cditpiAfhAw\"")
        os.system("tar -xf UTKFace.tar.gz")
        os.system("rm UTKFace.tar.gz")
        os.system("mv UTKFace " + job_dir + "/")
    age = []
    gender = []
    race = []
    img_path = []
    sizes = []
    for file in os.listdir(job_dir + "/UTKFace"):
        temp_size = size(os.path.getsize(job_dir + "/UTKFace/" + file))
        # Skip the low quality files
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


def group_age(data: pd.DataFrame) -> pd.Series:
    # Groups the ages into 14 categories and adds it to the dataframe
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

    # Create a list of the values we want to assign for each condition
    age_groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # Create a new column and use np.select to assign values to it using our lists as arguments
    return np.select(conditions, age_groups)


def get_data(job_dir: str = "..") -> pd.DataFrame:
    # Calls the other functions to get the data, read it into a dataframe, categorize the age, and return it
    data = read_dataset(job_dir)
    data["age_group"] = group_age(data)
    return data
