# Project: Follow Me

[//]: # (Image References)
[img_seg1]: ./images/segmentation1.png
[img_seg2]: ./images/segmentation2.png
[img_hard_to_see]: ./images/hard_to_see.png
[img_hard_to_see2]: ./images/hard_to_see2.png
[img_htsz1]: ./images/htsz1.png
[img_htsz2]: ./images/htsz2.png
[img_collecting_data]: ./images/training.png
[img_final_score]: ./images/final_score.png
[img_bumpy1]: ./images/bumpy1.png
[img_bumpy2]: ./images/bumpy2.png
[img_bumpy3]: ./images/bumpy3.png
[img_smooth1]: ./images/smooth1.png
[img_smooth2]: ./images/smooth2.png
[img_smooth3]: ./images/smooth3.png
[img_gpu_info]: ./images/GPU.png
[img_learning_curves]: ./images/learningrates.jpeg
[img_model_architecture]: ./images/model.png
[gif_introduction]: ./images/introduction.png
[jupyter_notebook]: ./code/model_training.ipynb


## Project Report

The goal of this project is to design and train a Fully Convolutional Network model to perform semantic segmentation and identify a specific person. The image recognition implemented is then used by a quadcopter to follow the target in a simulated environment.

![][gif_introduction]

This report briefly describes the Neural Network architecture used, the training process and results obtained.


## Model Architecture

Convolutional Neural Networks are a great tool to identify whether or not a specific pattern is present in an image. One limitation is that they cannot identify where within the image the searched object is located since they do not preserve spatial information. For this reason, Fully Convolutional Networks (FCN) were used as they can provide a segmentation of which pixels belong to the desired pattern and which do not belong.  

The main layers that were used are:
* Input layer with three features (RGB)
* Encoder with 3 layers and depths of 32, 64 and 128
* One 1x1 Convolution layer with depth of 264
* Decoder with 3 layers and depths of 32, 64 and 128


![][img_model_architecture]

The __encoder__ is used to extract features from the images. It is a series of convolutional layers where smaller patches are used to scan the input layer. By sharing the same parameters, it is computationally efficient and as a result, the layers can learn to recognize specific patterns regardless of its position within the original image. The complexity of the patterns it can learn to recognize is related to its depth.

As the dimensions of the images decrease during the convolutional operations, the information about the big picture can be lost. To preserve the information from multiple resolutions, __skip connections__ are used. They link non-adjacent layers with the same dimension and performing an element-wise addition operation. As a result, this technique help to make more precise segmentation decisions and improve accuracy.

 __1x1 convolutions__ are used to retain spatial information, making the neural network deeper and allowing it to learn more complex relationships. This layer keeps the same dimensions and is computationally cheap since it consists of regular matrix multiplication.

The __decoder__ uses transposed convolutions to restore dimensions and provide a pixel-wise segmentation. The mathematical operation is the inverse of the convolutional layers. It upscales the output of the encoder layer to generate results with the same size of the original image.

To improve segmentation results it is recommended building __deeper__ layers rather than wider. Considering computation resources and low resolution of the dataset images, 3 layers for each encoder and decoder were used in this project.


## Training the network

__Data__

The quality and amount of data used to train the network are extremely important to achieve good results. In this project, sample data was provided and additional data can be collected using the simulator and setting predetermined paths for the quadcopter, hero, and people spawn points.

![][img_collecting_data]

__Environment__

The model was implemented in ![this][jupyter_notebook]  jupyter notebook and trained in an AWS instance. The machine has a Tesla K80 GPU that makes the training processing much faster than in standard computers.

![][img_gpu_info]

__Model Parameters__

The __learning rate__ and the __epoch number__ are key parameters to be defined. Decreasing the learning rate makes training slower but can achieve better scores. The epoch number must be large enough to allow the training to achieve a good score but cannot be too large otherwise the model can become overfitted for the training set.

A higher __number of workers__ accelerates the training but are restricted by the processing resources available and the batch size. For batch sizes of 100, the workers were set to 2. If the batches were decreased to 50, it was possible to train using 4 workers.

The __batch size__ and __number of steps__ should be selected based on the training data size to guarantee everything will be processed once every epoch. They are also highly dependent on the amount of available memory.

__Training__

Several runs were necessary to explore the results of changing the parameters. Due to the high computation demand, using the AWS was crucial, especially for the long runs. 


One challenge was to tune each hyper-parameter isolated to better visualize their influence. For instance, the expected influence of the learning rate in the loss are widely known in the field and summarized in the following image

![][img_learning_curves]

*image credits: http://cs231n.github.io/neural-networks-3/*




To obtain smooth curves in this project, it was important to select the batch size and number of steps based on the data size. (e.g. `batch_size * steps_per_epoch =  data_size`)
Unfortunately, for big epoch numbers such as `300` the training time was significatly long (`>6h`) and the generated model achieved final scores under 0.40.

Graph            |  Parameters      | Comments |
:---------------:|:----------------:|:--------:|
![][img_smooth1] | ``` learning_rate = 0.001; batch_size = 32; num_epochs = 20; steps_per_epoch = 150; validation_steps = 50; workers = 4; final_score = 0.37 ``` | High learning rate |
![][img_smooth2] | ``` learning_rate = 0.00005; batch_size = 100; num_epochs = 300; steps_per_epoch = 42; validation_steps = 14; workers = 2; final_score = 0.33 ``` | Good learning curve, but score not high enough |
![][img_smooth3] | ``` learning_rate = 0.00001; batch_size = 100; num_epochs = 300; steps_per_epoch = 42; validation_steps = 14; workers = 2; final_score = 0.27; ``` | Slow learning rate, increasing the epochs could get good results |


Interestingly, faster and better final scores were achieved when the batch size and number of steps based were not matching the data size. (e.g. `batch_size * steps_per_epoch >  data_size`).
In those scenarios, the loss curves had several spikes and the relationship with the learning rate was harder to visualize.


Graph            |  Parameters      |
:---------------:|:----------------:|
![][img_bumpy1]  | (not recorded)    |
![][img_bumpy2]  | ```learning_rate = 0.005; batch_size = 128; num_epochs = 50; steps_per_epoch = 100; validation_steps = 50; workers = 2; final_score = 0.41``` |

One hypothesis that could justify these results is that the selected batch size and number of steps forced each epoch to be trained with different data sets, explaining the spikes in validation loss. Occasionally the trained model provided good results specifically for images used in the scoring. This hypothesis has not been verified and must be further investigated.

Unexpectedly, using additional data to train the model did not present significant improvements in final results. It could be caused by the quality of the data collected. Maybe not random enough or covering scenarios that did not match with the images used in the final scoring.

## Final Results

In order to achieve project requirements, the following parameters were used:
```shell
learning_rate = 0.001
batch_size = 100
num_epochs = 10
steps_per_epoch = 166
validation_steps = 12
workers = 2
```

The originally provided data was used to train the model
```shell
ubuntu@ip-xx:~/RoboND-DeepLearning-Project/data$ find train/images/ | wc -l 
4132
ubuntu@ip-xx:~/RoboND-DeepLearning-Project/data$ find validation/images | wc -l 
1185

```

The model file can be found [here](data/weights/model_weights) and it is saved in 
```shel
ubuntu@ip-xx:~/RoboND-DeepLearning-Project/data/weights$ file model_weights 
model_weights: Hierarchical Data Format (version 5) data

```


The final score calculated based on IoU was 0.41
![][img_final_score]

With this model it is possible to perform semantic segmentation to identify the target successfully in most of the cases.
![][img_seg2]
![][img_seg1] 

But when the targe is far away from the camera, it fails to recognize it (me too)
![][img_hard_to_see]
![][img_hard_to_see2]
 
Testing in the simulator, the model successfully identified and followed the target.
```shell
python follower.py --pred_viz model_weights
```

## Conclusions and future enhancements

The use of Keras makes it easier and faster to implement the network but as abstracts several details of the underlying architecture. To make specific customizations it would be more appropriate to implement the architecture directly using TensorFlow. 

The FNC architecture implemented is rather simple and pretty straight forward. Yet, it is powerful and very flexible. The very same architecture could be used to recognize other objects or animals, but a new model would have to be trained using additional data properly labeled.

To improve results, one simple change would be to increase input layer resolution. The failure to identify the target when it is far from the camera has a significant impact on the final score but even for the human eye, it is impossible to recognize it with current image resolution.

Dropout is a powerful and widely-used technique to be tested in this project. It randomly discards propagations making the network redundant and more robust. Other options include using a larger network with early termination to prevent overfitting and use adaptive learning rate to optimize the learning curve.





