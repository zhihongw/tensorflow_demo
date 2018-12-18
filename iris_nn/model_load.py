import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
    with tf.Session as sess:
        tf.saved_model.loader.load(
            sess,
            [],
        './models',
        )
        prediction = graph.get_tensor_by_name('')

        sess.run(prediction, feed_dict={
        })