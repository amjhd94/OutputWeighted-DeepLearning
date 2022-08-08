import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from utils import custom_KDE
import time
sns.set()
np.random.seed(10)
tf.keras.backend.set_floatx('float64')

#%% =========================================== %%#
##### Stochastic signal (SDE) #####

np.random.seed(10)

sigma = 2.  # Standard deviation.
mu = 2.  # Mean.
tau = .5  # Time constant.

dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.

sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)

x = np.zeros(n)

for i in range(n - 1):
    x[i + 1] = x[i] + dt * (-(.25*x[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()

y = np.atleast_2d(x).T
x = t

plt.figure()
plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('y')

## =========================================== ##
#%% Creating training dataset

training_set = y[:int(.8*len(y)), :]
test_set = y[int(.8*len(y)):, :]
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

history_window = 10
prediction_window = 1
stride_window = 1

x_train = []
y_train = []
for i in range(0, len(training_set_scaled) - history_window - prediction_window + 1, stride_window):
    x_train.append(training_set_scaled[i:(i+history_window)])
    y_train.append(training_set_scaled[(i+history_window):(i+history_window+prediction_window)])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

y_train = np.reshape(y_train, (len(x_train), prediction_window))

true_pdf = custom_KDE(y_train, bw=.01)
true_pdf_y, true_pdf_py = true_pdf.evaluate()
pdf_x = y_train
pdf_y = np.interp(pdf_x, true_pdf_y, true_pdf_py)
pdf_y = pdf_y*(pdf_y>=0)

pdf_y_train = np.reshape(pdf_y, (len(y_train), 1))

plt.figure()
plt.subplot(1,2,1)
plt.semilogy(true_pdf_y, true_pdf_py, 'k')
plt.subplot(1,2,2)
plt.plot(y_train, '-');plt.plot(1/pdf_y, '')

#%% randomize
rand_idx = np.random.permutation(x_train.shape[0])
X_train = x_train[rand_idx,:]
Y_train = y_train[rand_idx,:]
pdf_Y_train = pdf_y_train[rand_idx,:]


# Prepare the training dataset.
batch_size = int(len(X_train)/10)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train, pdf_Y_train)).batch(batch_size)

#%% Simple DNN-LSTM model

model = keras.Sequential()
model.add(layers.Dense(4, input_shape=(X_train.shape[1],1)))
model.add(layers.Dense(8))
model.add(layers.Dense(16))
model.add(layers.LSTM(units=16))
model.add(layers.Dense(16))
model.add(layers.Dense(8))
model.add(layers.Dense(4))
model.add(layers.Dense(units=prediction_window))

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-2)

# Prepare the metrics.
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

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
loss_type = loss_types[0]

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
    
#%% Creating and scaling test data

inputs  = sc.transform(test_set)

# Preparing X_test and predicting the prices
X_test = []
for i in range(0, len(test_set) - history_window - prediction_window + 1, stride_window):
    X_test.append(inputs[i:i+history_window,:])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],int(history_window),1))
y_test = model.predict(X_test)
y_test = sc.inverse_transform(y_test)

#%%

plt.figure()
plt.semilogy(np.array(tr_loss))
plt.title('Training loss')

plt.figure()
plt.subplot(1,2,1)
plt.plot(test_set[history_window+1:], '-', label='True')
plt.plot(y_test, '-r', label='Model')
plt.legend()
plt.title('Test_pred')

pred_pdf = custom_KDE(model.predict(X_test), bw=.01)
pred_pdf_y, pred_pdf_py = pred_pdf.evaluate()

true_pdf = custom_KDE(inputs[history_window+1:], bw=.01)
true_pdf_y, true_pdf_py = true_pdf.evaluate()

plt.subplot(1,2,2)
plt.semilogy(true_pdf_y, true_pdf_py, 'k', label='True')
plt.semilogy(pred_pdf_y, pred_pdf_py, 'r', label='Model')
plt.legend()


