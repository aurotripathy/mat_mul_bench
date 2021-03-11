# Mostly modified from https://github.com/yaroslavvb/stuff/blob/master/matmul_benchmark.py
# 16K x 16K matric mul
# http://stackoverflow.com/questions/41804380/testing-gpu-with-tensorflow-matrix-multiplication

import os
import sys
import time
import argparse

import tensorflow.compat.v1 as tf  # original code based on TF V1
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--tf32', help="tf32 execution",
                    action='store_true')
parser.add_argument('--precision', '-p', help="pick data type",
                    type=str, required=True, choices=['fp32', 'fp16'])

args = parser.parse_args()

try:
    if args.tf32:
        tf.config.experimental.enable_tensor_float_32_execution(True)
        print('enabling tf32')
    else:
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print('disabling tf32')
except AttributeError as error:
    print('Non Fatal ERROR:', error)

n = 8192 * 2
if args.precision == 'fp32':
    dtype = tf.float32
else:
    dtype = tf.float16
    
with tf.device("/gpu:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)


# avoid optimizing away redundant nodes
config = tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(optimizer_options=tf.compat.v1.OptimizerOptions(opt_level=tf.compat.v1.OptimizerOptions.L0)))
sess = tf.compat.v1.Session(config=config)

sess.run(tf.compat.v1.global_variables_initializer())
iters = 10

# warming up
sess.run(product.op)

start = time.time()
for i in range(iters):
  sess.run(product.op)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**12

print(f'TF32:{args.tf32} Precision:{args.precision} Dim:{n} x {n} matmul took: {elapsed/iters:.5f} sec, {rate:.2f} T flops')



