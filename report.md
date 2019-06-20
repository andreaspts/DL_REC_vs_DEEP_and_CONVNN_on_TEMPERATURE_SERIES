# Analysis report



https://www.deeplearningbook.org/contents/rnn.html

## Introduction

In the following we seek to apply a simple fully connected network, a convnet and different recurrent neural networks to analyze a weather timeseries dataset (the so-called "Jena climate dataset") and predict the air temperature 24 hours in the future. An excerpt of the time series is depicted in the subsequent plot:

<p align="center">
  <img width="460" height="300" src="temperature_series_excerpt.png">
</p>

Generally speaking, the goal is to feed the networks with the training set,....

The underlying insight to use deep neural networks for such a task (essentially a fitting-to-data problem) is that they are very powerful [function approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

In our image classification task we want to compare the performance on timeseries prediction of a [fully connected](https://www.deeplearningbook.org/contents/mlp.html), a [convolutional](https://www.deeplearningbook.org/contents/convnets.html) and a [recurrent](https://www.deeplearningbook.org/contents/rnn.html) neural network. 

The basic difference in between a fully connected and a convolutional layer is that the former learns global patterns in their input feature space while the latter learns local patterns. Importantly, these patterns are translation invariant and once a specific pattern is learnt in a part of an image it can be recognized in another part. In this way, convnets need less training examples as they have a greater generalization power. In addition, convnets can learn more and more complex visual concepts while going deeper into the network. Hence, they can recognize statistical hierarchies of patterns.  


### Network architectures (aspects of the learning algorithm)

#### Dense network:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 3360)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                107552    
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 107,585
Trainable params: 107,585
Non-trainable params: 0
```


#### GRU network:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru_3 (GRU)                  (None, 32)                4512      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 33        
=================================================================
Total params: 4,545
Trainable params: 4,545
Non-trainable params: 0
```

#### GRU network (bidirectional):

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 64)                9024      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 65        
=================================================================
Total params: 9,089
Trainable params: 9,089
Non-trainable params: 0
```


#### GRU network with dropout:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru_4 (GRU)                  (None, 32)                4512      
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 33        
=================================================================
Total params: 4,545
Trainable params: 4,545
Non-trainable params: 0
```

#### Stacked GRU network with dropout:


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru_5 (GRU)                  (None, None, 32)          4512      
_________________________________________________________________
gru_6 (GRU)                  (None, 64)                18624     
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 65        
=================================================================
Total params: 23,201
Trainable params: 23,201
Non-trainable params: 0
```


#### CONVNET:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, None, 32)          2272      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, None, 32)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, None, 32)          5152      
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, None, 32)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, None, 32)          5152      
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 12,609
Trainable params: 12,609
Non-trainable params: 0
```


#### CONVNET and GRU network:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_4 (Conv1D)            (None, None, 32)          2272      
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, None, 32)          0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, None, 32)          5152      
_________________________________________________________________
gru_1 (GRU)                  (None, 32)                6240      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 13,697
Trainable params: 13,697
Non-trainable params: 0
```

## Observation and results

First run:

keras.optimizers.RMSprop
lr 0.01

<p align="center">
  <img width="760" height="400" src="plot2.png">
</p>

<p align="center">
  <img width="760" height="400" src="plot1.png">
</p>




GRU, GRU dropout, GRU stacked
Epoch 11/20
500/500 [==============================] - 238s 476ms/step - loss: nan - val_loss: nan

exploding gradients problem (the likely cause of the nans)



Possible reasons:

    Gradient blow up
    Your input contains nan (or unexpected values)
    Loss function not implemented properly
    Numerical instability in the Deep learning framework

You can check whether it always becomes nan when fed with a particular input or is it completely random.

Usual practice is to reduce the learning rate in step manner after every few iterations.


attempt at solution: L2 regularization added. does not lead to a significant improvement (at most a few more episodes before nan)

another attempt: fix the issue indirectly rather than directly. I would recommend using gradient clipping, which will simply clip any gradients that are above a certain value. 

Second run:

keras.optimizers.Adam
lr 0.005

kernel_regularizer=regularizers.l2(0.001)

clipnorm=1.0


<p align="center">
  <img width="760" height="400" src="plot3.png">
</p>

<p align="center">
  <img width="760" height="400" src="plot4.png">
</p>




### Improvements


#### Regularization: dropout, weight regularization and batch normalization
