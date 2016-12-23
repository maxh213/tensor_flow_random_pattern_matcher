import tensorflow as tf
from random import *

TRAINING_DATA_AMOUNT = 1000
IMPORTANT_ELEMENTS = [False, False, False, False, True, False, True, False, False, False]
LEARNING_RATE = 0.03
INPUT_SIZE = len(IMPORTANT_ELEMENTS)
OUTPUT_SIZE = 2 #2 output states: true or false
LAYER_1_WEIGHTS = INPUT_SIZE * 2
LAYER_2_WEIGHTS = INPUT_SIZE


TRUE_OUTPUT = [1, 0]
FALSE_OUTPUT = [0, 1]

TRAINING_DATA_AMOUNT = 100
TRAINING_LOOPS = 100000

randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]

def get_weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def get_bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def train():
	input = tf.placeholder(tf.float32, [None, INPUT_SIZE])
	output = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
	global_step = tf.Variable(0, trainable=False)

	layer1_weights = get_weight_variable([INPUT_SIZE, LAYER_1_WEIGHTS])
	layer1_bias = get_bias_variable([LAYER_1_WEIGHTS])

	layer1_output = tf.nn.softmax(tf.nn.tanh(tf.matmul(input, layer1_weights) + layer1_bias))

	layer2_weights = get_weight_variable([LAYER_1_WEIGHTS, LAYER_2_WEIGHTS])
	layer2_bias = get_bias_variable([LAYER_2_WEIGHTS])

	layer2_output = tf.nn.softmax(tf.nn.tanh(tf.matmul(layer1_output, layer2_weights) + layer2_bias))

	layer3_weights = get_weight_variable([LAYER_2_WEIGHTS, OUTPUT_SIZE])
	layer3_bias = get_bias_variable([OUTPUT_SIZE])

	layer3_output = tf.nn.softmax(tf.nn.tanh(tf.matmul(layer2_output, layer3_weights) + layer3_bias))

	tf_output = layer3_output
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_output, output))

	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step=global_step)

	correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(tf_output),1), tf.argmax(output,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()

	sess.run(tf.initialize_all_variables())

	print("Training beginning")
	for i in range(TRAINING_LOOPS):
		inputs, outputs = get_data(TRAINING_DATA_AMOUNT)
		entropy, _, train_step_accuracy = sess.run([cross_entropy,train_step, accuracy], feed_dict={input: inputs, output: outputs})
		if i % 50 == 0:
			print("Training loop " + str(i) + "/" + str(TRAINING_LOOPS))
			print("Entropy: " + str(entropy))
			print("Training Step Result Accuracy: " + str(train_step_accuracy))
			if train_step_accuracy == 1.0:
				break	
	print("Training ended")

	print("Testing...")

	test_inputs, test_outputs = get_data(TRAINING_DATA_AMOUNT)
	print("Testing Accuracy: " + str(sess.run(accuracy, feed_dict={input: test_inputs, output: test_outputs})))

def get_data(amount):
	input = []
	output = []
	for i in range(amount):
		new_input = randBinList(INPUT_SIZE)
		new_output = get_output_for_input(new_input)
		input.append(new_input)
		output.append(new_output)
	return input, output


def get_output_for_input(input):
	for i in range(len(input)):
		if IMPORTANT_ELEMENTS[i] == True:
			if input[i] == 1:
				return TRUE_OUTPUT
	return FALSE_OUTPUT 

if __name__ == '__main__':
	train()
