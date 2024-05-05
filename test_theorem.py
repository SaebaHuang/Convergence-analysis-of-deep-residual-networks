# encoding=UTF-8
import numpy as np


def constraint(t, axis, ref_value):
    t_abs = tf.abs(t)
    if axis is not None:
        t_abs = tf.reduce_sum(t_abs, keepdims=True, axis=axis,)
    desired = tf.clip_by_value(t_abs, clip_value_min=0.0, clip_value_max=ref_value)
    clipped_tensor = t * desired / (t_abs + tf.keras.backend.epsilon())
    return clipped_tensor


def compute_norm(t, axis=None):
    if axis is None:
        axis = [0, 1, 2]
    return tf.sqrt(tf.reduce_sum(tf.square(t), axis=axis))


if __name__ == '__main__':
    # freely allocate memory
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # fix random seed
    from numpy.random import seed
    seed(321)
    tf.random.set_seed(321)

    # randomly generate data points
    num_hidden_channels = 128
    data_points = tf.random.uniform(shape=[1000, 32, 32, 128], minval=0.0, maxval=1.0, dtype=tf.float32)

    # standard convolution
    from numpy import float_power, single, inf, sqrt
    std_conv_out = data_points

    # residual blocks
    num_residual_blocks = 500
    depth_residual_blocks = 2
    max_val = 1.0
    gamma = 0.2
    diff_list = []
    is_building_block = True
    network_out = std_conv_out
    for i in range(num_residual_blocks):
        copy_network_out = network_out
        filter_scale = single(float_power(i+1, (1+gamma)/depth_residual_blocks))
        bias_scale = single(float_power(i+1, 1+gamma))

        if not is_building_block:
            cur_filter = tf.random.uniform(shape=[1, 1, num_hidden_channels, num_hidden_channels//4], minval=-max_val / sqrt(num_hidden_channels),
                                           maxval=max_val / sqrt(num_hidden_channels)) / filter_scale
            cur_bias = tf.random.uniform(shape=[num_hidden_channels//4], minval=-max_val, maxval=max_val) / bias_scale
            network_out = tf.nn.conv2d(network_out, cur_filter, strides=[1, 1, 1, 1], padding='SAME') + cur_bias
            network_out = tf.nn.relu(network_out)

            # obtain residual block output
            for _ in range(depth_residual_blocks - 2):
                cur_filter = tf.random.uniform(shape=[1, 1, num_hidden_channels//4, num_hidden_channels//4], minval=-max_val * 2.0 / (3 * sqrt(num_hidden_channels)), maxval=max_val * 2.0 / (3 * sqrt(num_hidden_channels))) / filter_scale
                cur_bias = tf.random.uniform(shape=[num_hidden_channels//4], minval=-max_val, maxval=max_val) / bias_scale
                network_out = tf.nn.conv2d(network_out, cur_filter, strides=[1, 1, 1, 1], padding='SAME') + cur_bias
                network_out = tf.nn.relu(network_out)

            cur_filter = tf.random.uniform(shape=[1, 1, num_hidden_channels//4, num_hidden_channels], minval=-max_val * 2.0 / sqrt(num_hidden_channels), maxval=max_val * 2.0 / sqrt(num_hidden_channels)) / filter_scale
            cur_bias = tf.random.uniform(shape=[num_hidden_channels], minval=-max_val, maxval=max_val,) / bias_scale
            network_out = tf.nn.conv2d(network_out, cur_filter, strides=[1, 1, 1, 1], padding='SAME') + cur_bias
        else:
            for _ in range(depth_residual_blocks - 1):
                cur_filter = tf.random.uniform(shape=[3, 3, num_hidden_channels, num_hidden_channels],
                                               minval=-max_val,
                                               maxval=max_val)
                cur_filter = tf.clip_by_norm(cur_filter, 1.0, axes=[0, 1, 2])
                cur_bias = tf.random.uniform(shape=[num_hidden_channels], minval=-max_val, maxval=max_val)
                network_out = tf.nn.conv2d(network_out, cur_filter / filter_scale, strides=[1, 1, 1, 1], padding='SAME') + cur_bias / bias_scale
                network_out = tf.nn.relu(network_out)

            cur_filter = tf.random.uniform(shape=[3, 3, num_hidden_channels, num_hidden_channels],
                                           minval=-max_val,
                                           maxval=max_val)
            cur_filter = cur_filter = tf.clip_by_norm(cur_filter, 1.0, axes=[0, 1, 2])
            cur_bias = tf.random.uniform(shape=[num_hidden_channels], minval=-max_val, maxval=max_val)
            network_out = tf.nn.conv2d(network_out, cur_filter / filter_scale, strides=[1, 1, 1, 1], padding='SAME') + cur_bias / bias_scale

        network_out = network_out + copy_network_out
        network_out = tf.nn.relu(network_out)

        cur_diff = tf.reduce_max(tf.abs(network_out - copy_network_out)).numpy()
        print('{} layer: {:.5f}'.format(i, cur_diff))
        diff_list.append(cur_diff)

    # plot the curve of diff
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 20})
    figure = plt.figure(figsize=(10, 10), dpi=400)
    ax = figure.add_subplot(1, 1, 1)
    plt.plot(np.arange(1, 1 + num_residual_blocks, 1), diff_list)
    ax.set_xlabel('Layers')
    ax.set_ylabel('Difference')
    plt.yscale('log')
    plt.minorticks_off()
    plt.savefig('./results/test_theorem/test_plot.pdf', format='pdf', dpi=400)
    plt.close(figure)





