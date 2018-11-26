from __future__ import print_function

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import os

from keras.callbacks import ModelCheckpoint
from utils import load_data, svm_classify
from linear_cca import linear_cca
from models import create_model
import matplotlib.pyplot as plt
from dataset import MMImdbDataset, report_performance


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train_model(model, data1, data2, epoch_num, batch_size):
    """
    trains the model
    # Arguments
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively. data should be packed
        like ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        epoch_num: number of epochs to train the model
        batch_size: the size of batches
    # Returns
        the trained model
    """

    # Unpacking the data
    train_set_x1, train_set_y1 = data1[0]
    valid_set_x1, valid_set_y1 = data1[1]
    test_set_x1, test_set_y1 = data1[2]

    train_set_x2, train_set_y2 = data2[0]
    valid_set_x2, valid_set_y2 = data2[1]
    test_set_x2, test_set_y2 = data2[2]

    # best weights are saved in "temp_weights.hdf5" during training
    # it is done to return the best model based on the validation loss
    checkpointer = ModelCheckpoint(filepath="temp_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)

    # if use loss implemented on Tensorflow
    # we need to throw away data that cannot form a whole batch
    batch_num_tr = len(train_set_x1) // batch_size
    batch_num_val = len(valid_set_x1) // batch_size
    batch_num_tt = len(test_set_x1) // batch_size

    num_samp_tr = batch_size * batch_num_tr
    num_samp_val = batch_size * batch_num_val
    num_samp_tt = batch_size * batch_num_tt

    train_set_x1 = train_set_x1[0:num_samp_tr, :]
    train_set_x2 = train_set_x2[0:num_samp_tr, :]

    valid_set_x1 = valid_set_x1[0:num_samp_val, :]
    valid_set_x2 = valid_set_x2[0:num_samp_val, :]

    test_set_x1 = test_set_x1[0:num_samp_tt, :]
    test_set_x2 = test_set_x2[0:num_samp_tt, :]

    # used dummy Y because labels are not used in the loss function
    history = model.fit([train_set_x1, train_set_x2], np.zeros(num_samp_tr),
                        batch_size=batch_size, epochs=epoch_num, shuffle=True,
                        validation_data=([valid_set_x1, valid_set_x2],
                                         np.zeros(num_samp_val)),
                        callbacks=[checkpointer])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.load_weights("temp_weights.h5")

    results = model.evaluate([test_set_x1, test_set_x2],
                             np.zeros(num_samp_tt), batch_size=batch_size, verbose=1)

    print('loss on test data: ', results)

    results = model.evaluate([valid_set_x1, valid_set_x2],
                             np.zeros(num_samp_val), batch_size=batch_size, verbose=1)
    print('loss on validation data: ', results)
    return model


def tt_model(model, data1, data2, outdim_size, apply_linear_cca):
    """produce the new features by using the trained model
    # Arguments
        model: the trained model
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively.
            Data should be packed like
            ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        outdim_size: dimension of new features
        apply_linear_cca: if to apply linear CCA on the new features
    # Returns
        new features packed like
            ((new X for train - view 1, new X for train - view 2, Y for train),
            (new X for validation - view 1, new X for validation - view 2, Y for validation),
            (new X for test - view 1, new X for test - view 2, Y for test))
    """

    # producing the new features
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k][0], data2[k][0]])
        r = int(pred_out.shape[1] / 2)
        new_data.append([pred_out[:, :r], pred_out[:, r:], data1[k][1]])

    # based on the DCCA paper, a linear CCA should be applied on the output of the networks because
    # the loss function actually estimates the correlation when a linear CCA is applied to the output of the networks
    # however it does not improve the performance significantly
    if apply_linear_cca:
        w = [None, None]
        m = [None, None]
        print("Linear CCA started!")
        w[0], w[1], m[0], m[1] = linear_cca(new_data[0][0], new_data[0][1], outdim_size)
        print("Linear CCA ended!")

        # Something done in the original MATLAB implementation of DCCA, do not know exactly why;)
        # it did not affect the performance significantly on the noisy MNIST dataset
        # s = np.sign(w[0][0,:])
        # s = s.reshape([1, -1]).repeat(w[0].shape[0], axis=0)
        # w[0] = w[0] * s
        # w[1] = w[1] * s
        ###

        for k in range(3):
            data_num = len(new_data[k][0])
            for v in range(2):
                new_data[k][v] -= m[v].reshape([1, -1]).repeat(data_num, axis=0)
                new_data[k][v] = np.dot(new_data[k][v], w[v])

    return new_data


if __name__ == '__main__':
    ############
    # Parameters Section

    # the path to save the final learned features
    save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 100

    # size of the input for view 1 and view 2
    input_shape1 = 300
    input_shape2 = 4096

    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]

    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 10
    batch_size = 800

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = True

    # end of parameters section
    ############

    data = {}
    for subset in ['train', 'dev', 'test']:
        ds = MMImdbDataset(which_sets=(subset,),
                           file_or_path='/var/www/html/mmimdb/multimodal_imdb.hdf5',
                           sources=('features', 'vgg_features', 'genres'))
        data[subset] = next(ds.create_stream().get_epoch_iterator())

    data1 = ((data['train'][0], data['train'][2]),
             (data['dev'][0], data['dev'][2]),
             (data['test'][0], data['test'][2]),
            )
    data2 = ((data['train'][1], data['train'][2]),
             (data['dev'][1], data['dev'][2]),
             (data['test'][1], data['test'][2]),
            )

    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/
    #data1 = load_data('noisymnist_view1.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz')
    #data2 = load_data('noisymnist_view2.gz', 'https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz')

    # Building, training, and producing the new features by DCCA
    model = create_model(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                         learning_rate, reg_par, outdim_size, use_all_singular_values, batch_size)
    model.summary()
    model = train_model(model, data1, data2, epoch_num, batch_size)
    new_data = tt_model(model, data1, data2, outdim_size, apply_linear_cca)

    # Training and testing of SVM with linear kernel on the view 1 with new features
    [test_acc, valid_acc], [test_p, valid_p], [test_label, valid_label] = svm_classify(new_data, C=0.01, view=1)
    print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    print("Accuracy on view 1 (test data) is:", test_acc * 100.0)
    report_performance(test_label, test_p, 0.5)

    # Training and testing of SVM with linear kernel on the view 2 with new features
    [test_acc, valid_acc], [test_p, valid_p], [test_label, valid_label] = svm_classify(new_data, C=0.01, view=2)
    print("Accuracy on view 2 (validation data) is:", valid_acc * 100.0)
    print("Accuracy on view 2 (test data) is:", test_acc * 100.0)
    report_performance(test_label, test_p, 0.5)

    # Saving new features in a gzip pickled file specified by save_to
    print('saving new features ...')
    f1 = gzip.open(save_to, 'wb')
    thepickle.dump(new_data, f1)
    f1.close()
