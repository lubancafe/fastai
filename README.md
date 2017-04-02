# Learn FAST.ai by trying, failing, digging

## The Core codes of lesson1 (Dogs & Cats) are 

<code>vgg = Vgg16() </code>

<code>batches = vgg.get_batches(path+'train', batch_size=batch_size) </code>

<code>val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2) </code>

<code>vgg.finetune(batches) </code>

<code>vgg.fit(batches, val_batches, nb_epoch=1)</code>

The running result: loss: 0.1229 - acc: 0.9672 - val_loss: 0.0581 - val_acc: 0.9820

## After Lesson1@fastai, dive into basics of deep learning
To solve such complex problem (image classification), you may feel above codes too simple just like me. 
This is the pain of top-down learning approach, the "whole-picture" is a black magic box. As we ask what's behind this black-box, we are asking a 'top-down' question in problem-solving way.

Questions I ask myself:
- How computer learns the weights and bias (W, b) of each pixel on an image set from training data? 
- And how the classification is calculated on valid data?
- Why softmax regression for image classification?

### Understand Deep Learning much easier - if you can read excel you can understand the learning process
Jeremy uses excel sheet to simulate deep learning on linear regression (y=Wx+b). The W and b are simple real value (y=2x + 30).
https://www.youtube.com/watch?v=qnoLMkosHuE&pbjreload=10&app=desktop

### [optional] Understand linear regression (y=Wx+b) with Machine Learning Basics on Udacity
https://classroom.udacity.com/courses/ud120/lessons/2301748537/concepts/24575785420923

### [recommended] Understand Deep Learning better with tensorflow tutorial-get_started
https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial
- Simple code with simple data. Learn from crafting the codes with very simple data set;
- Learning Reports, we can visualize the learning process with tensorboard.
As a result of learning y=Wx+b, W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11

https://www.tensorflow.org/get_started/mnist/beginners
MNIST problem solved with softmax regression. The pictures are very helpful for beginners.

My understanding on labels, weights, 


## Digg into VGG_16.py, there are more questions come out
<code>def VGG_16();</code>

<code>...</code>

<code>model.add(Dense(1000, activation='softmax'))</code>

<code>...</code>

### What is Activation Function in Deep Learning?
https://www.thoughtly.co/blog/deep-learning-lesson-2/

#### What is softmax?
<p><b>softmax</b> is a generalization of logistic function that “squashes”(maps) a K-dimensional vector z of arbitrary real values to a K-dimensional vector σ(z) of real values in the range (0, 1) that add up to 1.
e.g. [2.1, 3.4, 81, 13.5] => [0.021, 0.034, 0.81, 0.135]</p>

##### Why softmax?
https://www.quora.com/Artificial-Neural-Networks-Why-do-we-use-softmax-function-for-output-layer

#### What is relu?
https://www.quora.com/What-is-the-role-of-rectified-linear-ReLU-activation-function-in-CNN

#### What is Backpropagation?
http://colah.github.io/posts/2015-08-Backprop/
