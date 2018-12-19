import tensorflow as tf
graph = tf.Graph()
export_dir = "modelsv2"
with tf.Session(graph=graph) as sess:
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
  result = sess.run(y, feed_dict={x: data})
"""
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
"""
