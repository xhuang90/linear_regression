# -*- coding: utf-8 -*-
# @Time    : 2/4/21 5:34 PM
# @Author  : Hobey Wong
# @Contact : huangxinhong@cvte.com
# @Desc    :

"""
    CISC6930
"""

import pandas as pd
import numpy as np
from logs import logger
import matplotlib.pyplot as plt


def create_train_data():
    """

    @return:
    @rtype:
    """

    def create_new_dataset(num_of_data):
        """

        @param num_of_data:
        @type num_of_data: int
        @return: None
        @rtype: None
        """

        df_train = pd.read_csv('data/train-1000-100.csv')
        df_train_sub = df_train.head(num_of_data)

        # save new dataset
        file_out_name = 'train-{}(1000)-100'.format(num_of_data)
        df_train_sub.to_csv('data/{}.csv'.format(file_out_name), index=False)

        logger.info('{} is saved!'.format(file_out_name))

    for num in [50, 100, 150]:

        create_new_dataset(num_of_data=num)


def construct_x_y_matrix(file_name):
    """

    @param file_name: name of the file to process (without .csv)
    @type file_name: string
    @return:
    @rtype:
    """
    # STEP0. process dataset
    df_dataset = pd.read_csv('data/{}.csv'.format(file_name))
    df_dataset = df_dataset.dropna(how='any', axis=1)

    # STEP1. separate x val and target y
    df_x_val = df_dataset.drop(columns='y', axis=1)
    df_y_val = df_dataset['y']

    # STEP2. construct x matrix and y label
    # prepend a column with all values x0=1
    x_matrix = np.insert(df_x_val.values, 0, values=1, axis=1)
    y_vector = df_y_val.values

    return x_matrix, y_vector


def ordinary_least_square(x_arr, y, lbda):
    """
    pseudo-inverse X = inverse(np.dot(x.T, x)).dot(x.T)

    @param x_arr:
    @type x_arr:
    @param y:
    @type y:
    @param lbda:
    @type lbda:
    @return:
    @rtype:
    """

    i_matrix = np.eye(N=x_arr.shape[1])
    x_t_x = np.dot(x_arr.T, x_arr)
    xtx_penalty = x_t_x + lbda * i_matrix
    x_t_y = np.dot(x_arr.T, y)
    weight_matrix = np.dot(np.linalg.inv(xtx_penalty), x_t_y)

    return weight_matrix


def calculate_mse(y_label, y_pred):
    """

    @param y_label:
    @type y_label:
    @param y_pred:
    @type y_pred:
    @return:
    @rtype:
    """

    mse = (np.square(y_label - y_pred)).mean()

    return mse


def calculate_trn_test_mse(trn_file_name, test_file_name, lbda_lst):
    """

    @param trn_file_name:
    @type trn_file_name:
    @param test_file_name:
    @type test_file_name:
    @param lbda_lst:
    @type lbda_lst:
    @return:
    @rtype:
    """
    # get train & test data
    train_x, train_y = construct_x_y_matrix(file_name=trn_file_name)
    test_x, test_y = construct_x_y_matrix(file_name=test_file_name)

    train_mse_dict = {}
    test_mse_dict = {}

    for lbda in lbda_lst:
        # train model
        weight = ordinary_least_square(train_x, train_y, lbda=lbda)
        train_y_pred = np.dot(train_x, weight)
        train_mse = calculate_mse(y_label=train_y, y_pred=train_y_pred)
        # test
        test_y_pred = np.dot(test_x, weight)
        test_mse = calculate_mse(y_label=test_y, y_pred=test_y_pred)

        train_mse_dict[lbda] = train_mse
        test_mse_dict[lbda] = test_mse

        # logger.info('For dataset {} & {}, with λ = {}, we get test MSE = {} (train MSE = {}).'
        #             .format(trn_file_name, test_file_name, theta, test_mse, train_mse))

    return train_mse_dict, test_mse_dict


def get_min_test_mse(trn_file_name, test_file_name):
    """

    @param trn_file_name:
    @type trn_file_name:
    @param test_file_name:
    @type test_file_name:
    @return:
    @rtype:
    """

    lbda_lst = [i for i in range(0, 151)]
    # get train MSE and test MSE with different theta
    train_mse_dict, test_mse_dict = calculate_trn_test_mse(trn_file_name, test_file_name, lbda_lst)
    # get  minimum test MSE and corresponding theta value
    lbda_ = min(test_mse_dict, key=test_mse_dict.get)
    test_mse_ = round(test_mse_dict[lbda_], 4)
    # output
    logger.info('For dataset {} & {}, with λ = {}, we get min test MSE = {}.'
                .format(trn_file_name, test_file_name, lbda_, test_mse_))


def plot_mse(trn_file_name, test_file_name, min_lbda):
    """

    @param trn_file_name:
    @type trn_file_name:
    @param test_file_name:
    @type test_file_name:
    @param min_lbda:
    @type min_lbda:
    @return:
    @rtype:
    """
    # construct x axis
    lbda_lst = [i for i in range(min_lbda, 151)]
    # get train MSE and test MSE with different theta
    train_mse_dict, test_mse_dict = calculate_trn_test_mse(trn_file_name, test_file_name, lbda_lst)
    # construct y axis
    y_train_mse = list(train_mse_dict.values())
    y_test_mse = list(test_mse_dict.values())
    # plot theta and MSE
    fig = plt.figure()
    plt.plot(lbda_lst, y_train_mse, label='Train MSE')
    plt.plot(lbda_lst, y_test_mse, label='Test MSE')
    plt.legend(loc='best')
    plt.show()
    # then save image
    pic_name = 'tmp_res/{}_mse_lambda_from_{}'.format(trn_file_name, min_lbda)
    fig.savefig(pic_name + '.png')
    logger.info('{} is saved!'.format(pic_name))


def cross_validation(trn_file_name, test_file_name, k_fold=10):
    """

    @param trn_file_name:
    @type trn_file_name:
    @param test_file_name:
    @type test_file_name:
    @param k_fold:
    @type k_fold:
    @return:
    @rtype:
    """

    # get train & test data
    train_x, train_y = construct_x_y_matrix(file_name=trn_file_name)
    test_x, test_y = construct_x_y_matrix(file_name=test_file_name)

    train_x_len = len(train_x)
    subset_size = train_x_len / k_fold

    labd_mse_dict = {}
    for labd in range(0, 151):

        valid_mse = 0

        for i in range(k_fold):

            idx_begin = int(i * subset_size)
            idx_end = int((i + 1) * subset_size)

            validation_x = train_x[idx_begin: idx_end, :]
            validation_y = train_y[idx_begin: idx_end]
            train_train_x = np.concatenate((train_x[: idx_begin, :], train_x[idx_end:, :]), axis=0)
            train_train_y = np.concatenate((train_y[: idx_begin], train_y[idx_end:]), axis=0)

            # train model then get weight
            weight_mat = ordinary_least_square(train_train_x, train_train_y, labd)
            # predict y value
            y_valid_pred = np.dot(validation_x, weight_mat)
            valid_mse += calculate_mse(y_label=validation_y, y_pred=y_valid_pred)
        labd_mse_dict[labd] = valid_mse / k_fold

    print(labd_mse_dict)


if __name__ == '__main__':

    # create additional dataset
    # create_train_data()
    # train_dataset = ['train-100-10', 'train-100-100', 'train-1000-100',
    #                  'train-50(1000)-100', 'train-100(1000)-100', 'train-150(1000)-100']
    # test_dataset = ['test-100-10', 'test-100-100', 'test-1000-100',
    #                 'test-1000-100', 'test-1000-100', 'test-1000-100']
    # for trn, test in zip(train_dataset, test_dataset):
    #     # 1. For each of the 6 dataset, plot both the training set MSE
    #     # and the test set MSE as a function of (x-axis) in one graph.
    #     plot_mse(trn_file_name=trn, test_file_name=test, min_lbda=0)
    #     # 1.a For each dataset, which  value gives the least test set MSE?
    #     get_min_test_mse(trn_file_name=trn, test_file_name=test)
    #
    # # 1.b For each of dataset 100-100, 50(1000)-100, 100(1000)-100, provide an additional
    # # graph with ranging from 1 to 150.
    # for trn, test in zip(['train-100-100', 'train-50(1000)-100', 'train-100(1000)-100'],
    #                      ['test-100-100', 'test-1000-100', 'test-1000-100']):
    #     plot_mse(trn_file_name=trn, test_file_name=test, min_lbda=1)
    cross_validation(trn_file_name='train-100-100', test_file_name='test-100-100', k_fold=10)
