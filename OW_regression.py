import tensorflow as tf
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
#%% Simple DNN model

inputs = keras.Input(shape=(1,), name="input")
x = layers.Dense(4, activation="swish")(inputs)
x = layers.Dense(8, activation="swish")(x)
x = layers.Dense(16, activation="swish")(x)
x = layers.Dense(8, activation="swish")(x)
x = layers.Dense(4, activation="swish")(x)
outputs = layers.Dense(1, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Prepare the metrics.
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

#%% Constructing artificial data

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

true_pdf = custom_KDE(y, bw=.05)
true_pdf_y, true_pdf_py = true_pdf.evaluate()
pdf_x = y
pdf_y = np.interp(pdf_x, true_pdf_y, true_pdf_py)
pdf_y = pdf_y*(pdf_y>=0)

plt.figure()
plt.subplot(3,1,1)
plt.plot(x, y, '.')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(3,1,2)
plt.semilogy(true_pdf_y, true_pdf_py, 'k')
plt.xlabel('y')
plt.ylabel('p(y)')
plt.subplot(3,1,3)
plt.plot(x, 1/pdf_y, '.')
plt.xlabel('x')
plt.ylabel('1/p(y)')

#%% Prepare the training dataset.

batch_size = 256

x_train = np.reshape(x, (-1, 1))
y_train = np.reshape(y, (-1, 1))
pdf_y_train = np.reshape(pdf_y, (-1, 1))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, pdf_y_train)).batch(batch_size)

#%% Custom loss function, training and testing steps

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

@tf.function
def test_step(x, y):
    val_y_pred = model(x, training=False)
    val_acc_metric.update_state(y, val_y_pred )
#%% Training model

loss_types = ['MSE', 'PW', 'MIXED']
# ============= Select 0, 1 or 2 to select different train algorithms ============= #
loss_type = loss_types[2]

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
   
    
plt.figure()
plt.semilogy(np.array(tr_loss))
plt.xlabel('epoch')
plt.ylabel('Training loss')

y_test = model.predict(np.reshape(x_train, (-1, 1))) 

plt.figure()
plt.plot(x_train, y_train, '.')
plt.plot(x_train, y_test, '.r')

pred_pdf = custom_KDE(y_test, bw=.05)
pred_pdf_y, pred_pdf_py = pred_pdf.evaluate()

plt.figure()
plt.semilogy(true_pdf_y, true_pdf_py, 'k')
plt.semilogy(pred_pdf_y, pred_pdf_py, 'r')