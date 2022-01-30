<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Fine Tuning</div>
<div align="center"><img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/fine%20tune.gif?raw=true" width="80%"></div>



## Overview:
One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic feature maps to features associated specifically with the dataset.
In most convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple and generic features that generalize to almost all types of images. As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.
<br>
We have learned how to use a pre-trained Neural Network on a new dataset using so-called [Transfer Learning](https://github.com/Pradnya1208/Transfer-Learning), by re-routing the output of the original model just prior to its classification layers and instead use a new classifier that we had created. Because the original model was 'frozen' its weights could not be further optimized, so whatever had been learned by all the previous layers in the model, could not be fine-tuned to the new data-set.
This project shows how to do both Transfer Learning and Fine-Tuning using the Keras API for Tensorflow.

## Architecture:
For Transfer Learning we used the Inception v3 model but we will use the `VGG16` model in this tutorial because its architecture is easier to work with.
### Flowchart:
The idea is to re-use a pre-trained model, in this case the VGG16 model, which consists of several convolutional layers (actually blocks of multiple convolutional layers), followed by some fully-connected / dense layers and then a softmax output layer for the classification.

The dense layers are responsible for combining features from the convolutional layers and this helps in the final classification. So when the VGG16 model is used on another dataset we may have to replace all the dense layers. In this case we add another dense-layer and a dropout-layer to avoid overfitting.

The difference between Transfer Learning and Fine-Tuning is that in Transfer Learning we only optimize the weights of the new classification layers we have added, while we keep the weights of the original VGG16 model. In Fine-Tuning we optimize both the weights of the new classification layers we have added, as well as some or all of the layers from the VGG16 model.

<img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/vgg16.PNG?raw=true">
<br>

## Dataset:
[knifey-spoony dataset](https://github.com/Hvass-Labs/knifey-spoony)

## Implementation:

**libraries** : `matplotlib` `numpy` `pandas` `plotly` `sklearn` `keras` `tensorflow`

## Pre-trained model VGG16:
The following creates an instance of the pre-trained VGG16 model using the Keras API. This automatically downloads the required files if you don't have them already. Note how simple this is in Keras compared to Tutorial #08.

The VGG16 model contains a convolutional part and a fully-connected (or dense) part which is used for classification. If include_top=True then the whole VGG16 model is downloaded which is about 528 MB. If include_top=False then only the convolutional part of the VGG16 model is downloaded which is just 57 MB.

We will try and use the pre-trained model for predicting the class of some images in our new dataset, so we have to download the full model, but if you have a slow internet connection, then you can modify the code below to use the smaller pre-trained model without the classification layers.

```
model = VGG16(include_top=True, weights='imagenet')
```
## Data augmentation:
Keras uses a so-called data-generator for inputting data into the neural network, which will loop over the data for eternity.

We have a small training-set so it helps to artificially inflate its size by making various transformations to the images. We use a built-in data-generator that can make these random transformations. This is also called an augmented dataset.
```
datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')
```
We also need a data-generator for the test-set, but this should not do any transformations to the images because we want to know the exact classification accuracy on those specific images. So we just rescale the pixel-values so they are between 0.0 and 1.0 because this is expected by the VGG16 model.
```
datagen_test = ImageDataGenerator(rescale=1./255)
```
### Plot a few images:
<img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/fork.PNG?raw=true" width="50%">

## Class weigths:
The Knifey-Spoony dataset is quite imbalanced because it has few images of forks, more images of knives, and many more images of spoons. This can cause a problem during training because the neural network will be shown many more examples of spoons than forks, so it might become better at recognizing spoons.

Here we use scikit-learn to calculate weights that will properly balance the dataset. These weights are applied to the gradient for each image in the batch during training, so as to scale their influence on the overall gradient for the batch.

## Prediction using VGG16:
We can use the VGG16 model on a picture of a parrot which is classified as a macaw (a parrot species) with a fairly high score of 79%.
<img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/parrot.PNG?raw=true" width="30%">

<br>
We can then use the VGG16 model to predict the class of one of the images in our new training-set. The VGG16 model is very confused about this image and cannot make a good classification.
<img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/vgg16on%20fork.PNG?raw=true" width="70%">
<br>

## Transfer Learning:
The pre-trained VGG16 model was unable to classify images from the Knifey-Spoony dataset. The reason is perhaps that the VGG16 model was trained on the so-called ImageNet dataset which may not have contained many images of cutlery.

The lower layers of a Convolutional Neural Network can recognize many different shapes or features in an image. It is the last few fully-connected layers that combine these featuers into classification of a whole image. So we can try and re-route the output of the last convolutional layer of the VGG16 model to a new fully-connected neural network that we create for doing classification on the Knifey-Spoony dataset.

First we print a summary of the VGG16 model so we can see the names and types of its layers, as well as the shapes of the tensors flowing between the layers. This is one of the major reasons we are using the VGG16 model in this tutorial, because the Inception v3 model has so many layers that it is confusing when printed out.
```
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
```

We can see that the last convolutional layer is called 'block5_pool' so we use Keras to get a reference to that layer.
```
transfer_layer = model.get_layer('block5_pool')
```
We refer to this layer as the Transfer Layer because its output will be re-routed to our new fully-connected neural network which will do the classification for the Knifey-Spoony dataset.
<br>

Using the Keras API it is very simple to create a new model. First we take the part of the VGG16 model from its input-layer to the output of the transfer-layer. We may call this the convolutional model, because it consists of all the convolutional layers from the VGG16 model.

### Optimizer:
We use the Adam optimizer with a fairly low learning-rate. The learning-rate could perhaps be larger. But if you try and train more layers of the original VGG16 model, then the learning-rate should be quite low otherwise the pre-trained weights of the VGG16 model will be distorted and it will be unable to learn.
```
optimizer = Adam(lr=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
```
## Trainable Layers:
By default all the layers of the VGG16 model are trainable.
```
True:	input_1
True:	block1_conv1
True:	block1_conv2
True:	block1_pool
True:	block2_conv1
True:	block2_conv2
True:	block2_pool
True:	block3_conv1
True:	block3_conv2
True:	block3_conv3
True:	block3_pool
True:	block4_conv1
True:	block4_conv2
True:	block4_conv3
True:	block4_pool
True:	block5_conv1
True:	block5_conv2
True:	block5_conv3
True:	block5_pool
```
## Training and test accuracy:
<img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/train-test.PNG?raw=true" width="60%">

<br>

```
Test-set classification accuracy: 73.77%
```

```
Confusion matrix:
[[137   9   5]
 [ 56  81   0]
 [ 56  13 173]]
(0) forky
(1) knifey
(2) spoony
```
## Fine Tuning:
In Transfer Learning the original pre-trained model is locked or frozen during training of the new classifier. This ensures that the weights of the original VGG16 model will not change. One advantage of this, is that the training of the new classifier will not propagate large gradients back through the VGG16 model that may either distort its weights or cause overfitting to the new dataset.

But once the new classifier has been trained we can try and gently fine-tune some of the deeper layers in the VGG16 model as well. We call this Fine-Tuning.
<br>

We want to train the last two convolutional layers whose names contain 'block5' or 'block4'.
```
for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)
    
    # Set the layer's bool.
    layer.trainable = trainable
```

## Train and test accuracy:
<img src="https://github.com/Pradnya1208/Fine-Tuning/blob/main/readme_files/fine_tuning_traintest.PNG?raw=true" width="65%">

```
Test-set classification accuracy: 77.55%
```




### Learnings:
`transfer learning` `Fine Tuning` `VGG16`







## References:
[Transfer Learning and Fine Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning#fine_tuning)
[Transfer learning](https://github.com/Pradnya1208/Transfer-Learning)
### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner



[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]


