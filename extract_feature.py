import tensorflow as tf


def conv(inputs, output_dim, kernel_size=3, strides=1, padding='same'):
    return tf.layers.conv2d(inputs=inputs,
                            filters=output_dim,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer())


def max_pooling(inputs, pool_size=2, strides=2, padding='same'):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def res_block(inputs, output_dim, strides, training):
    input_dim = inputs.get_shape().as_list()[-1]

    if input_dim == output_dim:
        if strides == 2:
            shortcut = max_pooling(inputs)
        else:
            shortcut = inputs
    else:
        shortcut = conv(inputs, output_dim, 1, strides)

    conv1 = conv(inputs, output_dim, 3, strides)
    bn1 = tf.layers.batch_normalization(conv1, training=training)
    relu1 = tf.nn.relu(bn1)

    conv2 = conv(relu1, output_dim, 3)
    bn2 = tf.layers.batch_normalization(conv2, training=training)
    relu2 = tf.nn.relu(bn2)

    return tf.add(shortcut, relu2)


def extract_feature(images, training):
    # feature extracting network is implemented by vgg16 with resblock
    output_dim_list = [64, 128, 256, 512, 512]

    res1 = res_block(images, output_dim_list[0], 1, training)
    pool1 = max_pooling(res1)

    res2 = res_block(pool1, output_dim_list[1], 1, training)
    pool2 = max_pooling(res2)

    res3 = res_block(pool2, output_dim_list[2], 1, training)
    pool3 = max_pooling(res3)

    res4 = res_block(pool3, output_dim_list[3], 1, training)
    pool4 = max_pooling(res4)

    res5 = res_block(pool4, output_dim_list[4], 1, training)

    return res5  # [batch_size, image_width/16, image_height/16, 512]
