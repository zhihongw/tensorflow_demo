import tensorflow as tf
import pandas as pd
import numpy as np

seed = 101
tf.set_random_seed(seed)
np.random.seed(seed)

dataset = pd.read_csv("Iris_Dataset.csv")
dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding
values = list(dataset.columns.values)
labels = values[-3:]
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

serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'X_data': tf.FixedLenFeature(shape=[4], dtype=tf.float32),}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
X_data = tf.identity(tf_example['X_data'], name='X_data')  # use tf.identity() to assign name
#X_data = tf.placeholder(shape=[None, 4], dtype="float32", name="x")

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

class_tensor = tf.constant(labels)
table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor, default_value="UNKNOWN")

table_init = tf.tables_initializer()
sess.run(table_init)

values, indices = tf.nn.top_k(final_output, 3)
indices = tf.cast(indices, tf.int64)
classes = table.lookup(indices)

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
input_x=tf.saved_model.utils.build_tensor_info(serialized_tf_example)
output_y=tf.saved_model.utils.build_tensor_info(classes)
output_z=tf.saved_model.utils.build_tensor_info(values)
#prediction=tf.argmax(final_output, 1)

predict_iris=(
        tf.saved_model.signature_def_utils.build_signature_def(    
        inputs={
              tf.saved_model.signature_constants.CLASSIFY_INPUTS:input_x,
        },

        outputs={
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:output_y,
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:output_z, 
        },
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': input_x},
            outputs={'scores': output_z},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder=tf.saved_model.builder.SavedModelBuilder("./modelsv2/1")
builder.add_meta_graph_and_variables(
        sess,[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
             "predict_iris":predict_iris,
             'predict_click':prediction_signature,
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True,
        legacy_init_op=tf.saved_model.main_op.main_op())
builder.save()      
"""
# Prediction
print()
for i in range(len(X_test)):
    print('Actual:', y_test[i], 'Predicted:', np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))
"""

