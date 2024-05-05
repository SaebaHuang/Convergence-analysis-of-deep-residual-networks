# encoding=UTF-8

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

    # utils
    from utils import logger
    from os.path import exists
    from os import makedirs
    from time import strftime
    d = 75
    m = 128
    gamma = 0.2
    cur_timestamp = strftime('%Y_%m_%d_%H_%M_%S')
    result_dir = './results/result_d_{}_m_{}_{}/'.format(d, m, cur_timestamp)
    if not exists(result_dir):
        makedirs(result_dir)
    logger_dir = result_dir + 'result.txt'
    fit_hist_dir = result_dir + 'fit_hist.pkl'
    fit_result_dir = result_dir + 'fit_results.pdf'
    model_dir = result_dir + 'model_d_{}_m_{}'.format(d, m)
    logger('d: {:d}, m: {:d}, gamma: {:.5f}.'.format(d, m, gamma), logger_dir)

    # create model
    from model_ResNet import create_resnet_model
    from tensorflow.keras.optimizers import Adam

    resnet_model = create_resnet_model(
        fig_width=32,
        num_input_channels=3,
        num_classes=10,
        num_residual_blocks=d,
        depth_residual_blocks=2,
        num_hidden_channels=m,
        gamma=gamma,
        max_val=1.0,
        building_block=True
    )
    resnet_model.compile(
        optimizer=Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    resnet_model.summary()

    # Train
    from tensorflow.keras.callbacks import LearningRateScheduler

    def scheduler(epoch, lr):
        if epoch == 75:
            return lr * 0.1
        elif epoch == 50:
            return lr * 0.1
        else:
            return lr


    reduce_lr = LearningRateScheduler(scheduler, verbose=1)

    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import Callback
    class AdamLearningRateTracker(Callback):
        def on_epoch_end(self, epochs, logs=None):
            lr = K.eval(self.model.optimizer.lr)
            print('\nLR: {:.6f}\n'.format(lr))

    try:
        loss, acc = resnet_model.evaluate(
            X_train, Y_train_OH
        )
        logger('Initial Train loss: {:.5f}, Initial Train acc: {:.5f}'.format(loss, acc), logger_dir)
        loss, acc = resnet_model.evaluate(X_test, Y_test_OH)
        logger('Initial Test loss: {:.5f}, Initial Test acc: {:.5f}'.format(loss, acc), logger_dir)
        hist = resnet_model.fit(
            X_train, Y_train_OH,
            epochs=100, batch_size=128,
            verbose=1,
            callbacks=[reduce_lr],
            shuffle=True,
            validation_data=(X_test, Y_test_OH)
        )
    except Exception as e:
        print('Fit exception: {}'.format(e))
        # save current model
        resnet_model.save(model_dir)
        exit(-1)

    # save fit history
    import pickle
    with open(fit_hist_dir, 'wb') as f:
        pickle.dump(hist.history, f)

    # Plot fit results
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.savefig(fit_result_dir)
    plt.close(fig)

    # Evaluate
    loss, acc = resnet_model.evaluate(
        X_test, Y_test_OH
    )
    logger('Test loss: {:.5f}, Test acc: {:.5f}'.format(loss, acc), logger_dir)

    # save model
    resnet_model.save(model_dir)



