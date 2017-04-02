# Learn FAST.ai by trying, failing, digging
I am a developer learning fast.ai with no good math. 

# Lesson1 Dogs & Cats image classification

##### The Core codes of lesson1 are 

```python
vgg = Vgg16() 

batches = vgg.get_batches(path+'train', batch_size=batch_size) 
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)

vgg.finetune(batches) 
vgg.fit(batches, val_batches, nb_epoch=1)

```

The running result: loss: 0.1229 - acc: 0.9672 - val_loss: 0.0581 - val_acc: 0.9820

## After Lesson1@fastai, dive into basics of deep learning
To solve such complex problem (image classification), you may feel above codes too simple just like me. 
This is the pain of top-down learning approach, what behind "whole-picture" is a black magic box. As we ask what's behind this black-box, we are asking a 'top-down' question in problem-solving way.

Questions I ask myself:
- How computer learns the weights and bias (W, b) of each pixel on an image dataset from training data? [solved](https://github.com/lubancafe/fastai/blob/master/README.md#my-understanding-on-labels-weights-and-bias-y--wx--b)
- And how the classification is calculated on test data for predication? [solved](https://github.com/lubancafe/fastai/blob/master/README.md#my-understanding-on-labels-weights-and-bias-y--wx--b)
- Why softmax regression for image classification?

### Understand Deep Learning much easier - if you can read excel you can understand the learning process
Jeremy uses excel sheet to simulate deep learning on linear regression (y=Wx+b). The W and b are simple real value (2,30), y=2x + 30.
https://www.youtube.com/watch?v=qnoLMkosHuE&pbjreload=10&app=desktop

### [optional] Understand linear regression (y=Wx+b) with Machine Learning Basics on Udacity
https://classroom.udacity.com/courses/ud120/lessons/2301748537/concepts/24575785420923

### [recommended] Understand Deep Learning better with tensorflow tutorials
#### tensorflow get started
https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial
- Simple code with simple data. Learn by crafting the codes with very simple data set;
- Learning with reports, we can visualize the learning process with tensorboard.
As a result of learning y=Wx+b, W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11

#### MNIST with tensorflow for beginners
https://www.tensorflow.org/get_started/mnist/beginners
MNIST problem solved with softmax regression. The explanation pictures are very helpful for beginners. For example, below is softmax regression scalargraph,
![Image of softmax-regression](https://www.tensorflow.org/images/softmax-regression-scalargraph.png)

#### My understanding on labels, weights and bias (y = Wx + b):
Training Labels (mnist.train.labels) is [55000, 10] matrix of floats ;

Weights is [784, 10] matrix of floats (given a certain pixel and known number, what's the weight); while bias is [10] array of floats;

The training process is a process of calculating *unknown Weights and bias* by given *known lables on images*; pseudo equation as below,
```javascript
// train.y is known labels on images, a [55000, 10] matrix of float;
// train.x is known image pixel intensities, a [55000, 784] matrix of float;
// W is unknown weights, a [784, 10] matrix of float;
// b is unknown bias, a [10] array of float;

train.y = train.x * W (?) + b (?);

// after training, we will get W and b;
```

The classification is a process of calculating *unknown label* by given *trained weights and bias upon one image*. pseudo equation as below, 
```
// test.y is unknown label, a [10] array of float;
// test.x is known image pixel intensities, a [1, 784] matrix of float, or simply a [784] array of float;
// W is trained weights, a [784,10] matrix of float;
// b is trained bias, a [10] array of float;

test.y (?) = test.x * W + b 

// after calculating, we will get a [10] array of probabilities on classification (which number)
```

Here with visualization of trained weights by numbers [0,1,..,9]
![Image of weights](https://www.tensorflow.org/images/softmax-weights.png)

## Digg into VGG_16.py, questions come out

```
def VGG_16();

...

model.add(Dense(1000, activation='softmax'))

...

```

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
