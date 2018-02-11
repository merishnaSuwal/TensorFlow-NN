import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

#y=mx+b

y_true = (0.5*x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns= ['X DATA'])
y_df = pd.DataFrame(data=y_true, columns= ['Y'])
my_data = pd.concat([x_df, y_df], axis=1) #concatenate along the columns side by side

# my_data.sample(n=250).plot(kind='scatter', x='X DATA', y="Y")

#feed batches of data (batch by batch)
batch_size = 8

m = tf.Variable(0.81)
b = tf.Variable(0.17)

#placeholders

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

#define the operation/ graph

y_model = m * xph + b

#loss function

error = tf.reduce_sum(tf.square(yph-y_model)) #tf.square is the same as squaring

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000

    for i in range(batches):
        #choose the set of 8 random data for 1000 batches
        rand_index = np.random.randint(0, len(x_data), size= batch_size)
        feed = {xph: x_data[rand_index], yph: y_true[rand_index]}
        sess.run(train, feed_dict=feed)
    model_m, model_b = sess.run([m,b])
    print(model_m)
    print(model_b)

y_hat = x_data* model_m + model_b
# my_data.sample(n=250).plot(kind='scatter', x='X DATA', y="Y")
# plt.plot(x_data, y_hat, 'r')
# plt.show()

## ESTIMATOR API

feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)

#set up estimator inputs

# Can also do .pandas_input_fn
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)

eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)

## train the estimator

estimator.train(input_fn=input_func, steps=1000)

#evaluate

train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)

eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

# Predict

brand_new_data = np.linspace(0, 10, 10)

input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data}, shuffle=False)

predictions = []

for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

my_data.sample(n=250).plot(kind='scatter',x='X DATA',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'r')
plt.show()