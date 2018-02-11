import numpy as np
import tensorflow as tf
import pandas as pd
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
my_data.sample(n=250).plot(kind='scatter', x='X DATA', y="Y")
plt.plot(x_data, y_hat, 'r')
plt.show()