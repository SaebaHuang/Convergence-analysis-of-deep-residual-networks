# encoding=UTF-8
import numpy as np

if __name__ == '__main__':
    # free allocate
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # load dataset
    from data_loader import load_cifar10
    from tensorflow.keras.utils import to_categorical

    dataset_dir = './dataset/cifar10/'
    X_train, Y_train, X_test, Y_test = load_cifar10(dataset_path=dataset_dir)
    Y_train_OH = to_categorical(Y_train)
    Y_test_OH = to_categorical(Y_test)

    # result path
    result_path = 'result_d_75_m_128_2023_09_15_08_39_24'

    # load model
    from model_ResNet import load_resnet_model
    model_path = f'./results/{result_path}/model_d_75_m_128/'
    resnet_model = load_resnet_model(model_path)
    resnet_model.summary()
    standard_conv = resnet_model.get_layer('standard_conv')
    residual_blocks = resnet_model.get_layer('residual_blocks')
    filters = residual_blocks.residual_blocks_filters
    biases = residual_blocks.residual_blocks_biases

    A_list = [0.0]
    B_list = [0.0]
    C_list = [0.0]
    D_list = [0.0]
    A = 0.0
    B = 0.0
    gamma = 0.2
    depth_residual_block = 2
    from numpy import single, float_power
    for i in range(75):
        scale_filter = single(float_power(i + 1, (1.0 + gamma) / 2))
        scale_bias = single(float_power(i + 1, 1.0 + gamma))

        # compute A
        A_i = 1.0
        for j in range(depth_residual_block):
            cur_filter = filters[i][j]
            cur_norm = tf.reduce_sum(tf.abs(cur_filter), axis=[0, 1, 2, 3]).numpy() / scale_filter
            A_i *= cur_norm
        A += A_i / (3 * 3 * 128 * 128 * 128)
        A_list.append(A)
        C_list.append(C_list[-1] + 1.0 / scale_bias)
        D_list.append(D_list[-1] + 2.0 / scale_bias)

        # compute B
        B_i = 0.0
        for j in range(depth_residual_block):
            tmp = 1.0
            for jj in range(j+1, depth_residual_block):
                cur_filter = filters[i][jj]
                cur_norm = tf.reduce_sum(tf.abs(cur_filter), axis=[0, 1, 2, 3]).numpy() / scale_filter
                tmp *= cur_norm

            cur_bias = biases[i][j]
            cur_norm = tf.reduce_sum(tf.abs(cur_bias)).numpy() / scale_bias

            B_i += cur_norm * tmp
        B += B_i / (3 * 3 * 128 * 128 * 128)
        B_list.append(B)

    # load history
    import pickle

    with open(f'./results/{result_path}/fit_hist.pkl', 'rb') as fit_history_file:
        history = pickle.load(fit_history_file)

    print(np.argmax(history['accuracy']))
    print(np.max(history['accuracy']))
    print(np.argmax(history['val_accuracy']))
    print(np.max(history['val_accuracy']))
    print(history['val_accuracy'][-1])

    # Plot fit results
    base_dir = f'./results/{result_path}/'
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(1, 101)), history['loss'], label='Train loss')
    plt.plot(list(range(1, 101)), history['val_loss'], label='Test loss')
    plt.title('Losses during training')
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.title('Accuracies during training')
    plt.plot(list(range(1, 101)), history['accuracy'], label='Train accuracy')
    plt.plot(list(range(1, 101)), history['val_accuracy'], label='Test accuracy')
    plt.legend(loc='upper left')
    plt.savefig(f'{base_dir}loss_and_accuracy.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(0, 76)), A_list)
    plt.title(r'$R^{-2}\sum_{n=1}^{75}\left(\prod_{m=1}^{2}s^{(n,m)}\right)/{n^{1.2}}$')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(0, 76)), B_list)
    plt.title(r'$R^{-2}\sum_{n=1}^{75}\sum_{m=1}^{2}\left(P^{(n,m)}\|\mathbf{b}^{(n,m)}\|_{1}\right)/n^{2.4-0.6m}$')
    plt.savefig(f'{base_dir}comp.pdf')
    plt.close(fig)

