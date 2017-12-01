# A single neuron which can learn simple functions like AND, OR, etc.

import tensorflow as tf

x = [[[0.], [0.]], [[0.], [1.]], [[1.], [0.]], [[1.], [1.]]]
y = [0., 1., 1., 1.]

learning_rate = 0.01

x_input = tf.placeholder(dtype=tf.float32, shape=(2, 1))
y_input = tf.placeholder(dtype=tf.float32, shape=())

weight = tf.Variable(initial_value=tf.truncated_normal([1, 2]))
bias = tf.Variable(initial_value=tf.zeros([1, 1]))

neuron = tf.add(tf.matmul(weight, x_input), bias)
output = tf.sigmoid(neuron)

loss = tf.square(tf.subtract(y_input, output))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

initializer = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(initializer)

    total_loss = 1.

    while(total_loss > 0.0001):

        for i in range(4):
            sess.run(train, feed_dict={x_input: x[i], y_input: y[i]})

        total_loss = 0.

        for i in range(4):
            total_loss += sess.run(loss, feed_dict={x_input: x[i], y_input:
                                                    y[i]})

        print("Loss = " + str(total_loss))

    print()

    for i in range(4):
        print('Input: ' + str(x[i]))
        print('Actual: ' + str(y[i]))
        print('Predicted: ' + str(sess.run(output, feed_dict={x_input: x[i],
                                                              y_input: y[i]})))
        print()
