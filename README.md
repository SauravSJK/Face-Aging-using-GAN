# Face-Aging

A TensorFlow GAN model to transform input images based on target age. 

## Introduction

The ability to generate realistic images of a person's face as they age has many practical applications, from forensics to entertainment. However, face aging is a challenging problem due to the complex changes that occur in a person's facial features over time. In recent years, Generative Adversarial Networks (GANs) have emerged as a powerful tool for generating high-quality images. GANs are a type of deep neural network that consists of two components: a generator and a discriminator. The generator learns to generate images that are indistinguishable from real images, while the discriminator learns to distinguish between real and fake images. Together, these components form a game where the generator tries to fool the discriminator, and the discriminator tries to identify the fake images.

We propose a novel GAN architecture that is specifically designed for generating realistic images of a person's face as they age. Our model is trained on a large dataset of face images, and we evaluate its effectiveness in generating high-quality images. 

## Architecture

Our proposed novel GAN architecture consists of two generators, two age modulators, and a discriminator. One of the generators is responsible for aging/rejuvenating the input image to the target age while the other is responsible for converting the generated image back to the source age. The encoder-decoder architecture used for the generators is expected to encode the image such that only the identity information is present in the encoding. This encoding is then sent to an age modulator which also takes a target age group as input and generates age-specific features that are then sent to each layer of the decoder network for the generation of the aged/rejuvenated image. The same flow of data is followed by the second generator for the restoration of the original image. Once an image has been generated based on the target age group, the discriminator predicts the target age group. 

![Model Architecture](https://github.com/SauravSJK/Face-Aging/blob/a7c9bae1d1c8c47e7ccd54446adba2a3172d7029/Images/Architecture%20(Dark).png)

## Dataset

We have used the UTKFace dataset for training and validation. The UTKFace dataset is a large-scale face dataset that contains over 20,000 images of faces with annotations for age, gender, and ethnicity. The dataset includes images of faces in a wide range of ages, from 0 to 116 years old. To facilitate the use of the dataset for research purposes, the images are preprocessed to ensure that they are cropped and aligned consistently.

## Results

During the training process of our model, we observe a consistent pattern in the behavior of the loss values. Initially, the losses start with average values and exhibit a slight increase during the first few epochs. However, after this initial rise, the losses consistently decrease as the training progresses, ultimately converging to lower values by the end of the 100 epochs.

This pattern of decreasing losses indicates that our model is continuously improving its performance as it receives more training iterations. The fact that the losses consistently decrease suggests that the model is learning to generate more accurate and high-quality images of faces in the target age group.

To visualize this behavior, we can refer to the cumulative loss plot shown in the below figure. This plot provides a comprehensive view of the cumulative loss over the course of training. From the plot, we can observe a clear downward trend in the cumulative loss values, indicating the steady improvement of our model's performance.

![Cumulative Loss](https://github.com/SauravSJK/Face-Aging/blob/f9030ed3941dbdc63a981c2fe69c14f69dda5b64/Images/cumulative_loss.png)

The decreasing losses throughout the training process provide promising evidence that given more time and training iterations, our model has the potential to achieve even better results. This suggests that with additional training, the model could further refine its ability to generate realistic and age-appropriate facial images.

To provide a comprehensive evaluation, we compare these metrics with the results presented in our [reference paper](https://ieeexplore.ieee.org/document/9711081), as shown in the below table. Although our model's FID score falls short when compared to that of the reference paper, it achieves superior age prediction accuracy. This discrepancy suggests that while our generated images for the target age may not align with the distribution of the input images, as depicted in the below figure, the situation improves significantly when the images are transformed back to the source age. This improvement indicates that our model effectively controls the generation process, as evidenced by the favorable pixel reconstruction loss.

Metric|Method|Score
--|--|--
FID|RAGAN|57.78
FID|Ours (first generator)|336668.44
FID|Ours (second generator)|46282.770
Age Prediction Accuracy|RAGAN|61.405%
Age Prediction Accuracy|Ours (first generator)|95.7%
Age Prediction Accuracy|Ours (second generator)|85.8%

![Age Transformation Example](https://github.com/SauravSJK/Face-Aging/blob/ecd65929fc836deefd1434f0cc96ac142a7a4f54/Images/Transformations%20(Dark).png)

## Replication steps

To setup the environment for the model, execute the below:

1. Install virtualenv if not already installed

	`pip3 install virtualenv`

2. Create your new environment (called 'venv' here)

	`virtualenv venv`

3. Enter the virtual environment

	`source venv/bin/activate`
	
4. Clone the repository

	`git clone https://github.com/SauravSJK/Face-Aging.git`
	
5. Change directory

	`cd Face-Aging`

6. Install the requirements in the current environment

	`pip install -r requirements.txt`


To train the model from scratch, execute the below. This will:
1. Download the dataset
2. Create tfrecords
3. Train the discriminator and generator
4. Generate a prediction for the image "/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg" wth source age: 48

Note: This will take around 5 days to complete

	cd Code
	python3 main.py --strtopt "a"

To predict using trained models, execute the below.
The default image is "/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg" wth source age: 48. This can be changed by specifying a different image using the "prediction_file_name" argument.
The output file will be saved as "Result.jpg" in the Face_Aging directory.

	cd Code
	python3 main.py --strtopt "p" --prediction_file_name "/UTKFace/1_0_0_20161219200338012.jpg.chip.jpg"
