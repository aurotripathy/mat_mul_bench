# On Titan X (Pascal)
# 8192 x 8192 matmul took: 0.10 sec, 11304.59 G ops/sec
# http://stackoverflow.com/questions/41804380/testing-gpu-with-tensorflow-matrix-multiplication

import os
import sys
# import tensorflow as tf
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.config.experimental.enable_tensor_float_32_execution(True)

n = 8192 * 2
dtype = tf.float32
# dtype = tf.float16
with tf.device("/gpu:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)


# avoid optimizing away redundant nodes
# repacing with tf.compat.v1.ConfigProto
config = tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(optimizer_options=tf.compat.v1.OptimizerOptions(opt_level=tf.compat.v1.OptimizerOptions.L0)))
sess = tf.compat.v1.Session(config=config)

sess.run(tf.compat.v1.global_variables_initializer())
iters = 10

# pre-warming
sess.run(product.op)

start = time.time()
for i in range(iters):
  sess.run(product.op)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**12
print('\n %d x %d matmul took: %.5f sec, %.2f T flops/sec' % (n, n,
                                                            elapsed/iters, 
                                                            rate,))



