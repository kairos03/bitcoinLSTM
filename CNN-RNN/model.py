import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def depthwise_conv2d(x, W):
  return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='SAME')


def apply_depthwise_conv(x, kernel_size, num_features, depth):
  weights = weight_variable([1, kernel_size, num_features, depth])
  biases = bias_variable([depth * num_features])
  return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
  return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                        strides=[1, 1, stride_size, 1], padding='SAME')


def cnn():

  ### hyper parameter
  input_height = 1
  input_width = 30
  num_features = 8
  num_labels = 1

  batch_size = 10
  kernel_size = 20
  depth = 60
  num_hidden = 1000

  learning_rate = 0.001
  training_epochs = 20
  ###

  #feed
  X = tf.placeholder(name='X', shape=[None, input_height, input_width, num_features], dtype=tf.float32)
  Y = tf.placeholder(name='Y', shape=[None, num_labels], dtype=tf.float32)

  conv1 = apply_depthwise_conv(X, kernel_size=kernel_size, num_features=num_features, depth=depth)  #(?,1,30,8) -> (?,1,30,480)
  pool1 = apply_max_pool(conv1, 3, 3)   #(?,1,30,480) -> (?,1,10,480)
  conv2 = apply_depthwise_conv(pool1, kernel_size=)