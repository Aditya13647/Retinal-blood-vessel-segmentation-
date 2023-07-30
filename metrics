#################################################                        metrics code                                                ################################################# 


import tensorflow.keras.backend as K
import tensorflow as tf

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


auc = tf.keras.metrics.AUC()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = binary_crossentropy(y, y_pred)
        # Compute gradients
        grads = tape.gradient(loss, model.trainable_variables)
        # Update weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Update metrics
        auc.update_state(y, y_pred)
        sensitivity.update_state(y, y_pred)
        specificity.update_state(y, y_pred)
        # Return loss and metrics
        return loss, sensitivity.result(), specificity.result(), auc.result()
