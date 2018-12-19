import tensorflow as tf
import pandas as pd
import numpy as np

seed = 101
tf.set_random_seed(seed)
np.random.seed(seed)

dataset = pd.read_csv("Iris_Dataset.csv")
dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding
values = list(dataset.columns.values)
y = dataset[values[-3:]]
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype="float32")
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]
test_size = 10
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]

sess = tf.Session()
interval = 50
epoch = 500

X_data = tf.placeholder(shape=[None, 4], dtype="float32")
y_target = tf.placeholder(shape=[None, 3], dtype="float32")

hidden_layer_nodes = 8

w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes])) # Inputs -> Hidden Layer
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
b2 = tf.Variable(tf.random_normal(shape=[3]))

hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))
loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

print('Training the model...')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % interval == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))

#tf.saved_model.simple_save(sess, "./models",
#                           inputs = {"x": X_data },
#                           outputs = {"y": y_target})
saver_new = tf.train.Saver()
input_x=tf.saved_model.utils.build_tensor_info(X_data)
output_y=tf.saved_model.utils.build_tensor_info(y_target)

predict_iris=(
        tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': input_x},
        outputs={'y':output_y},
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
)
builder=tf.saved_model.builder.SavedModelBuilder("./modelsv2")
builder.add_meta_graph_and_variables(
        sess,[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
             "predict_iris":predict_iris,
        })
builder.save()      
"""
# Prediction
print()
for i in range(len(X_test)):
    print('Actual:', y_test[i], 'Predicted:', np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))
"""

