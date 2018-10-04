import tensorflow as tf

def pool(x):
    with tf.variable_scope('pool'):
        layer = tf.layers.average_pooling2d(
                inputs=x,
                pool_size=2,
                strides=2,
                padding='SAME',
                data_format='channels_last')
    return layer


def bn(x):
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
                training=True,
                trainable=True,
                name=None,
                reuse=None,
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

 
def conv1x1(x, out_channels, stride=1, padding=1, bias=False):
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
                reuse=None)
    return layer



def downsample(x, out_channels):
    '''
    The actual channel number is not downsampling, just to switch the channel number
    '''
    in_channels = x.shape[-1]
    if True:
        print('in_channels:', in_channels)
        print('out_channels:', out_channels)
    with tf.variable_scope('downsample'):
        bn1 = bn(x)
        relu1 = tf.nn.relu(bn1)
        conv1 = conv1x1(relu1, out_channels)
    return conv1


def conv_block(x, out_channels):
    with tf.variable_scope('convblock'):
        in_channels = x.shape[-1]
        with tf.variable_scope('conv1'):
            bn1 = bn(x)
            relu1 = tf.nn.relu(bn1)
            conv1 = conv3x3(relu1, int(out_channels / 2))
            
        with tf.variable_scope('conv2'):
            bn2 = bn(conv1)
            relu2 = tf.nn.relu(bn2)
            conv2 = conv3x3(relu2, int(out_channels / 4))
            
            
        with tf.variable_scope('conv3'):
            bn3 = bn(conv2)
            relu3 = tf.nn.relu(bn3)
            conv3 = conv3x3(relu3, int(out_channels / 4))

        out1 = tf.concat([conv1, conv2, conv3], axis=-1)
        if in_channels != out_channels:
            out2 = downsample(x, out_channels)
        else:
            out2 = x
        out3 = out1 + out2
    return out3


def bottleneck(x, out_channels, use_downsample=False):
    with tf.variable_scope('bottleneck'):
        conv1 = conv1x1(x, out_channels)
        bn1 = bn(conv1)
        relu1 = tf.nn.relu(bn1)

        conv2 = conv3x3(relu1, out_channels)
        bn2 = bn(conv2)
        relu2 = tf.nn.relu(bn2)

        conv3 = conv1x1(relu2, int(out_channels * 4))
        bn3 = bn(conv3)

        if use_downsample:
            residual = downsample(x, out_channels)
        else:
            residual = x

        out1 = bn3 + residual
        out2 = tf.nn.relu(out1)
    return out2


def hourglass(x, level):
    # upper branch
    with tf.variable_scope('up1'):
        up1 = conv_block(x, 256)
        
    # lower branch
    with tf.variable_scope('low1'):
        low1 = pool(x)
        print('low1 shape:', low1.shape)
        low1 = conv_block(low1, 256)

    with tf.variable_scope('low2'):
        if level > 1:
            low2 = hourglass(low1, level - 1)
        else:
            low2 = conv_block(low1, 256)

    with tf.variable_scope('low3'):
        low3 = conv_block(low2, 256)

    h = low3.shape[1]
    w = low3.shape[2]
    h = int(h * 2)
    w = int(w * 2)
    with tf.variable_scope('up2'):
        up2 = tf.image.resize_nearest_neighbor(low3, size=(h, w))

    out = up1 + up2
    return out


def fan(x, num_modules=1):
    with tf.variable_scope('fan'):
        # Base
        with tf.variable_scope('base'):
            with tf.variable_scope('conv1'):
                conv1 = conv7x7(x, 64)
                bn1 = bn(conv1)
                relu1 = tf.nn.relu(bn1)
            with tf.variable_scope('conv2'):
                conv2 = conv_block(relu1, 128)
                pool1 = pool(conv2)
            with tf.variable_scope('conv3'):
                conv3 = conv_block(pool1, 128)
            with tf.variable_scope('conv4'):
                conv4 = conv_block(conv3, 256)

        # Cat
        previous = conv4
        outputs = []
        for i in range(num_modules):
            with tf.variable_scope('hourglass' + str(i)):
                print('previous shape:', previous.shape)
                with tf.variable_scope('hg'+str(i)):
                    hg = hourglass(previous, 4)
                with tf.variable_scope('tmp_out'+str(i)):
                    ll = bn(hg)
                    ll = tf.nn.relu(ll)
                    tmp_out = conv1x1(ll, 68)
                    outputs.append(tmp_out)
                if i < num_modules - 1:
                    with tf.variable_scope('connector'+str(i)):
                        with tf.variable_scope('conv1'):
                            ll = conv1x1(ll, 256)
                        with tf.variable_scope('conv2'):
                            tmp_out_ = conv1x1(tmp_out, 256)
                        previous = previous + ll + tmp_out_
    return outputs



        
        
