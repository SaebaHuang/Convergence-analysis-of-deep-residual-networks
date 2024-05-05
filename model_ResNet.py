# encoding=UTF-8

import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, GlobalAveragePooling2D, Dense, ZeroPadding2D, RandomCrop, RandomFlip, RandomRotation
from tensorflow.keras.models import Model
from numpy import single, float_power
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.constraints import MinMaxNorm, MaxNorm, max_norm, UnitNorm
from numpy import sqrt
from tensorflow.keras import backend


class ElementConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self, ref_value, axis=None):
        self.ref_value = ref_value
        self.axis = axis

    def __call__(self, w):
        abs_val = tf.abs(w)
        if self.axis is not None:
            abs_val = tf.reduce_sum(abs_val, keepdims=True, axis=self.axis)
        desired = backend.clip(abs_val, 0.0, self.ref_value)
        return w * (desired / (abs_val + backend.epsilon()))

    def get_config(self):
        config_dict = super(ElementConstraint, self).get_config()
        config_dict['ref_value'] = self.ref_value
        config_dict['axis'] = self.axis
        return {'ref_value': self.ref_value, 'axis': self.axis}


class UnableZeroPadding2D(Layer):
    def __init__(self, padding, layer_name, **kwargs):
        super(UnableZeroPadding2D, self).__init__(name=layer_name)

        self.layer_name = layer_name
        self.padding = padding
        self.pad_layer = ZeroPadding2D(padding=padding, name='sub_pad')

    def call(self, inputs, training=False):
        if training:
            inputs = self.pad_layer(inputs)

        return inputs

    def get_config(self):
        config_dict = super(UnableZeroPadding2D, self).get_config()
        config_dict['layer_name'] = self.layer_name
        config_dict['padding'] = self.padding
        return config_dict


class ResNetLayer(Layer):
    def __init__(self, layer_name, num_residual_blocks, depth_residual_blocks, num_hidden_channels, gamma, max_val, building_block=False, **kwargs):
        super(ResNetLayer, self).__init__(name=layer_name)

        self.layer_name = layer_name
        self.num_residual_blocks = num_residual_blocks
        self.depth_residual_blocks = depth_residual_blocks
        self.num_hidden_channels = num_hidden_channels
        self.gamma = gamma
        self.max_val = max_val
        self.building_block = building_block

        self.residual_blocks_filters = None
        self.residual_blocks_biases = None
        self.scale_list = None
        self.in_channels = None

    def build(self, input_shape):
        super(ResNetLayer, self).build(input_shape)

        self.in_channels = input_shape[-1]
        self.residual_blocks_filters = []
        self.residual_blocks_biases = []

        for i in range(self.num_residual_blocks):
            if not self.building_block:
                # bottle neck block
                residual_block_filters = [
                    self.add_weight(
                        shape=(1, 1, self.in_channels, self.num_hidden_channels // 4),
                        constraint=max_norm(max_value=self.max_val, axis=[0, 1, 2]),
                        name='residual_block_{}_filter_{}'.format(i, 0)
                    )
                ] + [
                    self.add_weight(
                        shape=(3, 3, self.num_hidden_channels // 4, self.num_hidden_channels // 4),
                        constraint=max_norm(max_value=self.max_val, axis=[0, 1, 2]),
                        name='residual_block_{}_filter_{}'.format(i, j)
                    )
                    for j in range(1, self.depth_residual_blocks-1)
                ] + [
                    self.add_weight(
                        shape=(1, 1, self.num_hidden_channels // 4, self.in_channels),
                        constraint=max_norm(max_value=self.max_val, axis=[0, 1, 2]),
                        name='residual_block_{}_filter_{}'.format(i, self.depth_residual_blocks-1)
                    )
                ]

                residual_block_biases = [
                    self.add_weight(
                        shape=(self.num_hidden_channels // 4,),
                        initializer='zeros',
                        constraint=ElementConstraint(1.0),
                        name='residual_block_{}_bias_{}'.format(i, j)
                    )
                    for j in range(self.depth_residual_blocks-1)
                ] + [
                    self.add_weight(
                        shape=(self.in_channels,),
                        initializer='zeros',
                        constraint=ElementConstraint(1.0),
                        name='residual_block_{}_bias_{}'.format(i, self.depth_residual_blocks-1)
                    )
                ]
            else:
                # building blocks
                residual_block_filters = [
                    self.add_weight(
                        shape=(3, 3, self.num_hidden_channels, self.num_hidden_channels),
                        constraint=max_norm(self.max_val, axis=[0, 1, 2]),
                        name='residual_block_{}_filter_{}'.format(i, j)
                    )
                    for j in range(0, self.depth_residual_blocks)
                ]
                residual_block_biases = [
                    self.add_weight(
                        shape=(self.num_hidden_channels, ),
                        initializer='zeros',
                        constraint=ElementConstraint(self.max_val),
                        name='residual_block_{}_bias_{}'.format(i, j)
                    )
                    for j in range(0, self.depth_residual_blocks)
                ]           
                
            self.residual_blocks_filters.append(
                residual_block_filters
            )
            self.residual_blocks_biases.append(
                residual_block_biases
            )

        return

    def call(self, inputs, *args, **kwargs):

        out = inputs

        for i in range(self.num_residual_blocks):
            scale_filter = single(float_power(i+1, (1.0 + self.gamma)/self.depth_residual_blocks))
            scale_bias = single(float_power(i+1, 1.0 + self.gamma))
            old_out = out
            for j in range(self.depth_residual_blocks):
                # old
                out = tf.nn.conv2d(out, self.residual_blocks_filters[i][j] / scale_filter, strides=[1, 1, 1, 1], padding='SAME')
                out = out + self.residual_blocks_biases[i][j] / scale_bias
                  
                if j < self.depth_residual_blocks - 1:
                    out = tf.nn.relu(out)
            out = out + old_out
            out = tf.nn.relu(out)

        return out

    def get_config(self):
        config_dict = super(ResNetLayer, self).get_config()
        config_dict['layer_name'] = self.layer_name
        config_dict['num_residual_blocks'] = self.num_residual_blocks
        config_dict['depth_residual_blocks'] = self.depth_residual_blocks
        config_dict['num_hidden_channels'] = self.num_hidden_channels
        config_dict['gamma'] = self.gamma
        config_dict['max_val'] = self.max_val
        config_dict['building_block'] = self.building_block
        return config_dict


def create_test_unable_zero_padding_model():
    model_input = Input(shape=(32, 32, 3), name='input')
    model_output = UnableZeroPadding2D(padding=(2, 2), layer_name='Pad')(model_input)
    model = Model(inputs=model_input, outputs=model_output)
    return model


def create_resnet_model(fig_width, num_input_channels, num_classes, num_residual_blocks, depth_residual_blocks, num_hidden_channels, gamma, max_val, building_block):
    
    # Inputs
    model_input = Input(shape=(fig_width, fig_width, num_input_channels), name='model_input')
    
    # Augmentation module
    resized_input = UnableZeroPadding2D(padding=(2, 2), layer_name='Pad')(model_input)
    cropped_input = RandomCrop(height=32, width=32)(resized_input)
    flipped_input = RandomFlip(mode="horizontal")(cropped_input)
    rotated_input = RandomRotation(15.0/360)(flipped_input)
    augmented_input = rotated_input
    
    # Sampling
    standard_conv_output = Conv2D(
        filters=num_hidden_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=RandomNormal(mean=0.0, stddev=max_val / (3.0 * sqrt(num_hidden_channels))),
        trainable=True,
        name='standard_conv',
    )(augmented_input)
    
    # Residual blocks
    resnet_output = ResNetLayer(
        layer_name='residual_blocks',
        num_residual_blocks=num_residual_blocks,
        depth_residual_blocks=depth_residual_blocks,
        num_hidden_channels=num_hidden_channels,
        gamma=gamma,
        max_val=max_val,
        building_block=building_block
    )(standard_conv_output)
    
    # Average Pooling
    ap_output = GlobalAveragePooling2D(
    )(resnet_output)

    # FC
    model_output = Dense(
        units=num_classes, 
        activation='softmax',
        kernel_constraint=MinMaxNorm(min_value=64.0, max_value=1024.0, axis=0),
        trainable=True
    )(ap_output)
    
    model = Model(inputs=model_input, outputs=model_output)
    return model


def load_resnet_model(model_dir):
    from keras.models import load_model
    model = load_model(model_dir, custom_objects={'ResNetLayer': ResNetLayer, 'ElementConstraint': ElementConstraint})
    return model


if __name__ == '__main__':
    # free allocate
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    data_points = tf.random.uniform(shape=[1, 32, 32, 3], minval=0.0, maxval=1.0, dtype=tf.float32)
    test_model = create_test_unable_zero_padding_model()
    test_model.summary()
    x_1 = test_model.predict(data_points)
    x_2 = test_model(data_points, training=True)

    print(x_1.shape)
    print(x_2.shape)


