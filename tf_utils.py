import tensorflow as tf
### input the scope wrapper functio here for easy reuse....

def fc_model(x,y,img_size_flat=28*28,num_classes=10):
    '''
    Using tf.layer.Dense create a 3 - layered fc model with
    output sizes: 50,20,10. The last layer being the output
    class labels.
    Try:
        - uising different activations - relu,sigmoid,etc.
        - initializing the bias vectors to 1.0
        - different random initializations for the kernels
            refer to: https://www.tensorflow.org/api_docs/python/tf/keras/initializers

    for details on the layer refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    '''
    y_pred,y_pred_cls,logits = None, None, None
    return y_pred,y_pred_cls,logits

def convolutional_model(x,y,img_size_flat=28*28,num_classes=10):
    '''
    Using tf.layer.conv2d create a 4-layer convolutional model.
    With a final fc layer
    with each layer.
    Try:
        - Different kernel sizes
        - adding max pooling layers into the model
        - different padding types
        - remove bias from the convolutional filters

    for details on the layer refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/conv2d
    '''

    y_pred,y_pred_cls,logits = None, None, None
    return y_pred,y_pred_cls,logits

def lstm_model(x,y,img_size_flat=28*28,num_classes=10):
    '''
    Using the class tf.layer.LSTM create an LSTM model by
    treating each row as a single input in a sequence
    Try:
        - Different number of units
        - Treat each column as an input
    for details on the layer refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
    '''

    y_pred,y_pred_cls,logits = None, None, None
    return y_pred,y_pred_cls,logits

def fc_custom_model(x,y,layers = [50,50,50], img_size_flat=28*28,num_classes=10):
    '''
    Using tf.layer.Dense create a n-layered fc model.
    input:
    layers -> defines the output size of each layer.

    for details on the layer refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    '''
    y_pred,y_pred_cls,logits = None, None, None
    for l in layers:
        pass

    return y_pred,y_pred_cls,logits

def simple_model(x,y,img_size_flat=28*28,num_classes=10):
    # define model parameters
    weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))

    # create linear layer
    x_flat = tf.reshape(x,[tf.shape(x)[0],img_size_flat])
    logits = tf.matmul(x_flat, weights) + biases
    # add a non-linearity
    y_pred = tf.nn.softmax(logits)
    # just to have our final readable result
    y_pred_cls = tf.argmax(y_pred, axis=1)
    return y_pred,y_pred_cls,logits



def reconstruction_loss():
    pass


def custom_cross_entropy():
    pass

def cross_entropy_loss(y_true_cls,logits,num_classes=10):
    y_true = tf.one_hot(y_true_cls,depth = num_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                               labels=y_true)
    return tf.reduce_mean(cross_entropy)
