import tensorflow as tf

def pool(x):
    """a functional 2x2 average pooling
    """
    with tf.variable_scope('pool'):
        layer = tf.layers.average_pooling2d(
                inputs=x,
                pool_size=2,
                strides=2,
                padding='SAME',
                data_format='channels_last')
    return layer


class BatchNorm(object):
    def __init__(self):
        self.bn = tf.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                beta_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001, scope=None),
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                renorm=False,
                renorm_clipping=None,
                renorm_momentum=0.99,
                fused=None,
                trainable=True,
                virtual_batch_size=None,
                adjustment=None,
                name=None)

    def __call__(self, x, training):
        return self.bn(x, training=training)


def bn(x, training, name=None, reuse=None):
    with tf.variable_scope('batchnorm'):
        layer = tf.layers.batch_normalization(
                x,
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                beta_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001, scope=None),
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                training=training,
                trainable=True,
                name=name,
                reuse=reuse,
                renorm=False,
                renorm_clipping=None,
                renorm_momentum=0.99,
                fused=None,
                virtual_batch_size=None,
                adjustment=None)
    return layer


def conv7x7(x, out_channels, stride=2, padding=3, bias=False):
    with tf.variable_scope('conv7x'):
        layer = tf.layers.conv2d(
                inputs=x,
                filters=out_channels,
                kernel_size=7,
                strides=stride,
                padding='SAME',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=bias,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001, scope=None),
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                reuse=None)
    return layer


def conv3x3(x, out_channels, stride=1, padding=1, bias=False):
    with tf.variable_scope('conv3x'):
        layer = tf.layers.conv2d(
                inputs=x,
                filters=out_channels,
                kernel_size=3,
                strides=stride,
                padding='SAME',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=bias,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001, scope=None),
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                reuse=None)
    return layer


def conv1x1(x, out_channels, stride=1, padding=1, bias=False, reuse=None):
    with tf.variable_scope('conv1x'):
        layer = tf.layers.conv2d(
                inputs=x,
                filters=out_channels,
                kernel_size=1,
                strides=stride,
                padding='SAME',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=None,
                use_bias=bias,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001, scope=None),
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                reuse=reuse)
    return layer



def downsample(x, out_channels, training):
    '''
    The actual channel number is not downsampling, just to switch the channel number
    '''
    in_channels = x.shape[-1]
    if True:
        print('in_channels:', in_channels)
        print('out_channels:', out_channels)
    with tf.variable_scope('downsample'):
        bn1 = bn(x, training=training)
        relu1 = tf.nn.relu(bn1)
        conv1 = conv1x1(relu1, out_channels)
    return conv1


def conv_block(x, out_channels, training):
    with tf.variable_scope('convblock'):
        in_channels = x.shape[-1]
        with tf.variable_scope('conv1'):
            bn1 = bn(x, training=training)
            relu1 = tf.nn.relu(bn1)
            conv1 = conv3x3(relu1, int(out_channels / 2))

        with tf.variable_scope('conv2'):
            bn2 = bn(conv1, training=training)
            relu2 = tf.nn.relu(bn2)
            conv2 = conv3x3(relu2, int(out_channels / 4))


        with tf.variable_scope('conv3'):
            bn3 = bn(conv2, training=training)
            relu3 = tf.nn.relu(bn3)
            conv3 = conv3x3(relu3, int(out_channels / 4))

        out1 = tf.concat([conv1, conv2, conv3], axis=-1)
        if in_channels != out_channels:
            out2 = downsample(x, out_channels, training=training)
        else:
            out2 = x
        out3 = out1 + out2
    return out3

"""
def bottleneck(x, out_channels, use_downsample=False, training=False):
    with tf.variable_scope('bottleneck'):
        conv1 = conv1x1(x, out_channels)
        bn1 = bn(conv1, training=training)
        relu1 = tf.nn.relu(bn1)

        conv2 = conv3x3(relu1, out_channels)
        bn2 = bn(conv2, training=training)
        relu2 = tf.nn.relu(bn2)

        conv3 = conv1x1(relu2, int(out_channels * 4))
        bn3 = bn(conv3, training=training)

        if use_downsample:
            residual = downsample(x, out_channels, training=training)
        else:
            residual = x

        out1 = bn3 + residual
        out2 = tf.nn.relu(out1)
    return out2
"""

def hourglass(x, level, training):
    # upper branch
    with tf.variable_scope('up1'):
        up1 = conv_block(x, 128, training=training)

    # lower branch
    with tf.variable_scope('low1'):
        low1 = pool(x)
        print('low1 shape:', low1.shape)
        low1 = conv_block(low1, 128, training=training)

    with tf.variable_scope('low2'):
        if level > 1:
            low2 = hourglass(low1, level - 1, training=training)
        else:
            low2 = conv_block(low1, 128, training=training)

    with tf.variable_scope('low3'):
        low3 = conv_block(low2, 128, training=training)

    h = low3.shape[1]
    w = low3.shape[2]
    h = int(h * 2)
    w = int(w * 2)
    with tf.variable_scope('up2'):
        # upsampling
        up2 = tf.image.resize_nearest_neighbor(low3, size=(h, w))


    out = up1 + up2
    return out


def fan(x, num_modules=1, reuse=None, training=True):
    '''
    x NHWC image tensor, where H = 256, W = 256, C = 3, dtype= tf.float32, range from [-1, 1]
    '''
    x = tf.identity(x, name='image_tensor')
    x_dy, x_dx = tf.image.image_gradients(x)
    shape = tf.shape(x)
    shape = shape[1:3]

    x = tf.concat([x, x_dx, x_dy], axis=3, name='stacked_gradients')

    with tf.variable_scope('fan', reuse=reuse):
        # Base
        # feature extractor
        with tf.variable_scope('base'):
            with tf.variable_scope('conv1'):
                conv1 = conv7x7(x, 32)
                bn1 = bn(conv1, training=training)
                relu1 = tf.nn.relu(bn1)
            with tf.variable_scope('conv2'):
                conv2 = conv_block(relu1, 64, training=training)
                pool1 = pool(conv2)
            with tf.variable_scope('conv3'):
                conv3 = conv_block(pool1, 96, training=training)
            with tf.variable_scope('conv4'):
                conv4 = conv_block(conv3, 128, training=training)

        # Concatenated modules
        previous = conv4
        outputs = []
        for i in range(num_modules):
            with tf.variable_scope('hourglass' + str(i)):
                print('previous shape:', previous.shape)
                with tf.variable_scope('hg'+str(i)):
                    hg = hourglass(previous, 4, training=training)
                with tf.variable_scope('tmp_out'+str(i)):
                    ll = bn(hg, training=training)
                    ll = tf.nn.relu(ll)
                    tmp_out = conv1x1(ll, 68, reuse=reuse)
                    outputs.append(tmp_out)
                if i < num_modules - 1:
                    with tf.variable_scope('connector'+str(i)):
                        with tf.variable_scope('conv1'):
                            ll = conv1x1(ll, 128, reuse=reuse)
                        with tf.variable_scope('conv2'):
                            tmp_out_ = conv1x1(tmp_out, 128, reuse=reuse)
                        previous = previous + ll + tmp_out_
    resized_outputs = []
    for output in outputs:
        resized_output = tf.image.resize_images(output, shape, method=tf.image.ResizeMethod.BICUBIC)
        resized_output = tf.identity(resized_output, 'heatmap_tensor_' + str(len(outputs)))
        resized_outputs.append(resized_output)
    return resized_outputs





