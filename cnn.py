import tensorflow as tf 
import numpy as np 

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, 
        embedding_size=128, filter_sizes=[3, 4, 5], num_filters=128, l2_reg_lambda=0.0):
        
        #Placeholders
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #Create a convolution layer for each filter size
        conv_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            conv_op = self.conv_layer(self.embedded_chars_expanded, sequence_length, filter_size, embedding_size, num_filters)
            conv_outputs.append(conv_op)

        # Combine the outputs of all convolution layers 
        num_filters_total = num_filters * len(filter_sizes)
        self.conv_op_combined = tf.concat(conv_outputs, 3)
        self.conv_op_flattened = tf.reshape(self.conv_op_combined, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.conv_op_flattened, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        
    def conv_layer(self, input, sequence_length,filter_size, embedding_size, num_filters):
        with tf.name_scope("conv_layer"):
            filter_shape = [filter_size, embedding_size, 1, num_filters]

            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="B")

            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            act = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            op = tf.nn.max_pool(act, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],padding='VALID')

            return op