
#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with twitter sentiment data.

    #Author: Rabindra Nath Nandi
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

# from grpc.beta import implementations
import numpy
import tensorflow as tf

# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2
#from tensorflow_serving.example import mnist_input_data
from data_helpers import load_data
from data_helpers import batch_iter
import numpy as np

x, y, vocabulary, vocabulary_inv = load_data(1)
# print(x)  # ['2' '144' '1073' ..., '0' '0' '0']
# print(y)  # [0 1]
# print(vocabulary)  # 'breakpoints': '169715', 'shrill': '61929', '1day': '22983'
# print(vocabulary_inv)  # ['krystyn'], ['litracey'], ['failbringer']
# return