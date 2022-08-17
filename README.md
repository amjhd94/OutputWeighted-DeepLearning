# Output-weighted deep learning
Among the problems that are often encountered in deep learning problems are overfitting and underfitting. The issue in forecasting, regression and danger map identification problems is that the consistent rare and extreme events have to be model due to their significance. However, the naive approach of training the model for "long" times lead to overfitting which limits versatility and generalization of the end model. Therefore, an uncenventional deep learning algortim is needed that is able to train the model on rare and extreme events as well as the frequent events without the problem of overfitting, thereby creating a robust model that can be used for prediction. 

In this project, there two demo codes that implement output-weighted deep learning algorithm in a regression problem (`OW_regression.py`) and a stochsatic forecasting problem ('OW_SDE_forecast.py').

## Getting Started
The codes was written, run and tested by Spyder IDE version 4.2.5.
The following is the required packages for this project:
```bash
pip install numpy==1.21.2
pip install scipy==1.4.1
pip install tensorflow==2.9.0
pip install keras==2.9.0
pip install kdepy==1.1.0
pip install matplotlib==3.4.3
pip install seaborn==0.11.2
```
With the packages installed, the code is ready to run.

## Tutorial
I will be going over only the `OW_regression.py` demo code since the basic idea for both demo codes is similar. We consider a 1-dimensional regession problem that is noise-polluted and contains an outlier that needs to be modeled. The figure below depicts the aforementioned dataset:

<img src="https://user-images.githubusercontent.com/110791799/185227780-275f0c0a-cadc-43db-92ab-109f92bcdb5e.png" alt="orig_ds" width="500"/>

I will show the predictions of a typically trained deep neural network based on "**MSE**" loss function and compare it with that of a deep neural network trained based on an "**output weighted**" loss function that weighs errors with the "**inverse of the occurrence probability**" of their corresponding true values.


1- We first begin by importing the required modules:
```import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils import custom_KDE
sns.set()
np.random.seed(410)
tf.keras.backend.set_floatx('float64')
```

2- Next, we create the noisy dataset with a rare event as shown in the figure above:
```py
data_size = 2048
noise_size = 512

x = np.random.uniform(low=-1, high=1, size=(data_size, 1))
y = 0*.5*np.random.uniform(low=-1, high=1, size=(data_size, 1)) + np.cos(15*x)/(1*x**2+1)

rand_extreme_idx = np.random.randint(low=0, high=len(x)-1, size=(1,), dtype=int)
extreme_vals = 1/np.random.uniform(low=.2, high=.3, size=rand_extreme_idx.shape)

noise_idx = np.random.randint(low=0, high=len(x)-1, size=(noise_size,), dtype=int)
noise_vals = .3*np.random.uniform(low=-1, high=1, size=(noise_size, 1))

Sum = 0
for i in range(len(rand_extreme_idx)):
    Sum = Sum + extreme_vals[i]/(1000000*(x-x[rand_extreme_idx[i]])**2+1)
y = y + Sum
y[noise_idx] = y[noise_idx] + noise_vals
```

3- Next, we create the deep neural network model, set the learning parameter and the accuracy metrics to track:
```py
inputs = keras.Input(shape=(1,), name="input")
x = layers.Dense(4, activation="swish")(inputs)
x = layers.Dense(8, activation="swish")(x)
x = layers.Dense(16, activation="swish")(x)
x = layers.Dense(8, activation="swish")(x)
x = layers.Dense(4, activation="swish")(x)
outputs = layers.Dense(1, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)

train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()
```

4- This is the important step where we create the output-weighted loss function and training step. There are three different loss functions that can be used here. One is the typicall **mean swuared error**. The other is the output weighted loss function which weighs the prediction error of each true output value by the inverse of the probability of its occurrence. And the last loss function is the combination of the two previous loss functions. The custom loss function, 'loss_fn' is then fed to the 'train_step' function which in turn uses it in the forward pass and the back propagation to update the model weights accordingly.

```py
def loss_fn(y_pred, y_true, py, loss_type='MSE'):
    
    if loss_type == 'MSE':
        loss = tf.reduce_mean((y_pred - y_true)**2)
        
    elif loss_type == 'PW': # Probability weighted
        loss = tf.reduce_mean(((y_pred - y_true)**2)*(1/py))
        
    elif loss_type == 'MIXED':
        loss = tf.reduce_mean(((y_pred - y_true)**2)*(1+1/py))
        
    return loss

@tf.function
def train_step(x, y, py, loss_type='MSE'):
    
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y_pred, y, py, loss_type=loss_type)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, y_pred)
    
    return loss_value
```

5- Now, we train our model with the different training algorithms defined above. The trainings are done under the same conditions except for the training algorithms to offer better grounds for comparison.
```py
loss_types = ['MSE', 'PW', 'MIXED']
# ============= Select 0, 1 or 2 to select different train algorithms ============= #
loss_type = loss_types[1]

tr_loss = []

epochs = 3500
for epoch in range(epochs):
    
    if epoch%100==0:
        print("\nStart of epoch %d" % (epoch,))
        
    start_time = time.time()

    # Iterate over the batches of the dataset.
    counter = 0
    for step, (x_batch_train, y_batch_train, pdf_y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train, pdf_y_batch_train, loss_type=loss_type)

    
    if epoch%100==0:
        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    tr_loss.append(train_acc_metric.result())
    train_acc_metric.reset_states()
```

The results show that 

- the model trained based on the typical "**MSE**" loss function, even though, is not overfitting, completely ignores the outlier:

<img src="https://user-images.githubusercontent.com/110791799/185228623-01a4336f-69e6-4a8c-a268-b0252bd77c67.png" alt="orig_ds" width="500"/>

- the model trained based on the "**output-weighted**" loss function, not only is not overfitting but also successfully and accurately learns the outlier:

<img src="https://user-images.githubusercontent.com/110791799/185228195-504280de-5c3f-4fbe-b3b9-d8d23939fc28.png" alt="orig_ds" width="500"/>

The results above show that an output-weighted deep learning algorithm is capable of creating robust unbiased models that can fully learn anomalies, rare events and outliers.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
