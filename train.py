import tensorflow as tf 
import numpy as np 
from tensorflow.contrib import learn
import data_helpers
from cnn import TextCNN
import os
import time

#Parameter
batch_size = 64
num_epochs = 200
evaluate_every = 100

#Get the reviews
x_text, y = data_helpers.load_data_and_labels('imdbreviews/reviews.pos','imdbreviews/reviews.neg')

#Vectorize the text
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set (10%) for training
dev_sample_index = -1 * int(0.1 * float(len(y)))
x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(x_train.shape[1], y_train.shape[1], vocab_size=len(vocab_processor.vocabulary_))

        writer = tf.summary.FileWriter('logs')
        writer.add_graph(sess.graph)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #single training step
        def train_step(x_batch, y_batch):
    
            feed_dict = { cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 0.5 }
            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

        #single test step
        def test_step(x_batch, y_batch):
            
            feed_dict = { cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0 }
            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy],feed_dict)

            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                test_step(x_test, y_test)
                print("")

            if current_step % evaluate_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model to {}\n".format(path))

