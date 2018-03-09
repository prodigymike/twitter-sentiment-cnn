'''
Send text to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

import time

from argparse import ArgumentParser

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto

from data_helpers import batch_iter, load_data, string_to_int


def evaluate_sentence(sentence, vocabulary):
    """
    Translates a string to its equivalent in the integer vocabulary and feeds it
    to the network.
    Outputs result to stdout.
    """
    x_to_eval = string_to_int(sentence, vocabulary, max(len(_) for _ in x))
    result = sess.run(tf.argmax(network_out, 1),
                      feed_dict={data_in: x_to_eval,
                                 dropout_keep_prob: 1.0})
    unnorm_result = sess.run(network_out, feed_dict={data_in: x_to_eval,
                                                     dropout_keep_prob: 1.0})
    network_sentiment = 'POS' if result == 1 else 'NEG'
    log('Custom input evaluation:', network_sentiment)
    log('Actual output:', str(unnorm_result[0]))


def parse_args():
    parser = ArgumentParser(description="Request a TensorFlow server for a prediction on the text")
    parser.add_argument("-s", "--server",
                        dest="server",
                        default='0.0.0.0:9000',
                        help="prediction service host:port")
    parser.add_argument("-t", "--text",
                        dest="text",
                        default="Cigarettes are nasty!",
                        help="text string",)
    args = parser.parse_args()

    host, port = args.server.split(':')
    
    return host, port, args.text


def main():
    # parse command line arguments
    host, port, text = parse_args()

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    # with open(text, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
    # data = f.read()

    start = time.time()

    request = predict_pb2.PredictRequest()

    # Call GAN model to make prediction on the text
    request.model_spec.name = 'twitter-sentiment'
    request.model_spec.signature_name = 'predict_text'

    # MJC: Convert string to int64
    # Evaluate custom input
    if FLAGS.custom_input != '':
        log('Evaluating custom input:', FLAGS.custom_input)
        evaluate_sentence(FLAGS.custom_input, vocabulary)

    # request.inputs['text'].CopyFrom(make_tensor_proto(data, shape=[1]))
    request.inputs['text'].CopyFrom(make_tensor_proto(text, shape=[1]))
    request.inputs['dropout'].CopyFrom(make_tensor_proto(1.0,shape=[1,1]))

    result = stub.Predict(request, 60.0)  # 60 secs timeout

    end = time.time()
    time_diff = end - start

    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    main()
