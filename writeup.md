# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./hist_train.png "Visualization"
[image2]: ./hist_valid.png "Visualization"
[image3]: ./hist_test.png "Visualization"
[image4]: ./sign_color.png "Original"
[image5]: ./sign_gray.png "Grayscaling"
[image6]: ./web_imgs/small_1.jpg "Traffic Sign 1"
[image7]: ./web_imgs/small_2.jpg "Traffic Sign 2"
[image8]: ./web_imgs/small_3.jpg "Traffic Sign 3"
[image9]: ./web_imgs/small_4.jpg "Traffic Sign 4"
[image10]: ./web_imgs/small_5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AndiChris/project_2_traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing often each class is present inside the training, validation and test data sets. 

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. Since the accuracy seemed similar compared to the full RGB images, converting to grayscale reduces the complexity a lot and therefor increasing training performance.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]
![alt text][image5]

As a last step, I normalized the image data because neuronal nets tend to train faster on normalized data. As a quick normalization I simple calculated the new pixel value as followed: (pixel - 128) / 128.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Grayscale image                       | 
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten               | outputs 400                                   |
| Dropout               |                                               |
| Fully connected       | outputs 120, mean 0, stddev 0.1               |
| RELU                  |                                               |
| Fully connected       | outputs 84, mean 0, stddev 0.1                |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected       | outputs 43,  mean 0, stddev 0.1               |
| Softmax               |                                               |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch size of 128 and 20 epochs. For optimizing the loss-gradient an AdamOptimizer was chosen. After some test runs with the final model architecture a learning rate of 0.003 seemed to produce the best accuracy values.

Additionally to prevent over-optimizing on the training set I added L2 regularization (with parameter beta 0.01) to the loss calculation.

The dropout layers were used with a probability-parameter of 0.6.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 95.9%
* test set accuracy of 93.7%

An iterative approach was chosen:
* **What was the first architecture that was tried and why was it chosen?**  
I started with the LeNet architecture from the LeNet lab solution because this architecture is already known to perform well on this sort of problems.


* **What were some problems with the initial architecture?**  
Overfitting with a to low validation accuracy of ~89%


* **How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**  
First I introduced dropout layers because I wanted to reduce overfitting which was the case with my architecture at first. After some iterations of evaluating I went with two dropout layers located before the first and the last fully-connected layers.  
As a second method I added L2-regularization to the loss calculation. This also helped decrease overfitting.


* **Which parameters were tuned? How were they adjusted and why?**  
The dropout probability was set to 0.6. It was an iterative approach as different number and location of dropout layers seemed to have impact on which probability works well. But in general the values were always between 0.5 and 0.8.  
Additionally the learning rate was adjusted to 0.003. This was done after plotting of the loss over the epochs. The first run with the default rate of 0.001 seemed to have a to slow decrease in loss. A value much higher had a fast first decrease followed by a plateau. 0.003 seemed to have a good balance between first decrease and good improvement over all epochs.

* **What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**  
A convolution layer works very well for this sort of problem because it exploits statistical invariants. For the traffic sign problem for example it does not matter where the sign in the image is. Its still the same sign so the network does not have to learn all cases separately.  
A dropout layer helps a lot because it introduces some random constrain when training. This way the net is prevented to over-train on the training set which will lead to bad performance on new unseen data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first image might be impossible to classify because as I realized afterwards the pedestrian sign in the training set is a different one. It is a triangle and not round and the person on the sign is shown from the side.

The second image might also be difficult because I had to resize it to match 32x32. During this operation the proportions were altered, meaning that the shape is slightly distorted.

The third, forth and fifth should be easy to classify because there are good clean images showing each sign from the front.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians           | Ahead only                                    | 
| Stop                  | Road work                                     |
| 70 km/h               | 70 km/h                                       |
| Yield                 | Yield                                         |
| Priority Road         | Priority Road                                 |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. Compared to the accuracy on the test set of 93.7% this is not quite good. But I think with further tweaking this can be increased. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the model is not able to tell that this is a pedestrian sign. As mentioned above this is due to the different pedestrian sign in the training set. However interestingly it predicts with 19% a "Ahead only" sign. If one looks at this sign the similarities are obvious. Both are an round sign with some white straight figure in the middle. In one case its an arrow and in the other a walking person.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .195                  | Ahead only                                    | 
| .095                  | Turn left ahead                               |
| .093                  | Children crossing                             |
| .062                  | Go straight or right                          |
| .053                  | Turn right ahead                              |


For the second image the model is absolutely unsure. Even in the top 5 probabilities the stop sign is not predicted.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .037                  | Road work                                     | 
| .030                  | 20 km/h                                       |
| .029                  | 50 km/h                                       |
| .028                  | Right-of-way at the next intersection         |
| .019                  | Bicycle crossing                              |

The third image is predicted correctly but with quite low accuracy. But the first three signs are very simillar beeing different only in the first digit (i.e. 70, 30 or 20). So accumulated one could argue it predicted with 22.8% a speed limit sign.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .089                  | 70 km/h                                       | 
| .073                  | 30 km/h                                       |
| .066                  | 20 km/h                                       |
| .030                  | Turn right ahead                              |
| .026                  | Roundabout mandatory                          |

The forth image did best of all images. It was predicted with 73.8% correctly to be a yield sign.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .738                  | Yield                                         | 
| .253                  | Keep left                                     |
| .199                  | 50 km/h                                       |
| .191                  | No passing                                    |
| .177                  | Turn left ahead                               |

The fifth image was also predicted correctly but with a 27.5% probability. Could be better.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .275                  | Priority road                                 | 
| .094                  | No vehicles                                   |
| .092                  | 50 km/h                                       |
| .089                  | 100 km/h                                      |
| .056                  | Traffic signals                               |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


