# -*- coding: utf-8 -*-
# @Time    : 2/4/21 5:34 PM
# @Author  : Hobey Wong
# @Contact : huangxinhong@cvte.com
# @Desc    :

"""
https://www.cnblogs.com/orangecyh/p/11647825.html

Implement L2 regularized linear regression algorithm with  ranging from 0 to 150
(integers only). For each of the 6 dataset, plot both the training set MSE and the test
set MSE as a function of  (x-axis) in one graph.
"""

import pandas as pd
import numpy as np
from logs import logger
import matplotlib.pyplot as plt


def construct_x_y_matrix(file_name):
    """

    @param file_name:
    @type file_name:
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
    x = np.insert(df_x_val.values, 0, values=1, axis=1)
    y = df_y_val.values

    return x, y


def calculate_beta(x_arr, y, coefficient):
    """

    @param x_arr:
    @type x_arr:
    @param y:
    @type y:
    @param coefficient:
    @type coefficient:
    @return:
    @rtype:
    """

    i_matrix = np.eye(N=x_arr.shape[1])
    x_t_x = np.dot(x_arr.T, x_arr)
    xtx_penalty = x_t_x + coefficient * i_matrix
    x_t_y = np.dot(x_arr.T, y)
    beta_matrix = np.dot(np.linalg.inv(xtx_penalty), x_t_y)

    return beta_matrix


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


def calculate_trn_test_mse(trn_file_name, test_file_name):
    """

    @param trn_file_name:
    @type trn_file_name:
    @param test_file_name:
    @type test_file_name:
    @return:
    @rtype:
    """
    # get train & test data
    train_x, train_y = construct_x_y_matrix(file_name=trn_file_name)
    test_x, test_y = construct_x_y_matrix(file_name=test_file_name)

    train_mse_dict = {}
    test_mse_dict = {}

    for theta in range(0, 151):
        # train model
        weight = calculate_beta(train_x, train_y, coefficient=theta)
        train_y_pred = np.dot(train_x, weight)
        train_mse = calculate_mse(y_label=train_y, y_pred=train_y_pred)

        test_y_pred = np.dot(test_x, weight)
        test_mse = calculate_mse(y_label=test_y, y_pred=test_y_pred)

        train_mse_dict[theta] = train_mse
        test_mse_dict[theta] = test_mse

        # logger.info('For dataset {} & {}, with lambda = {}, we get test MSE = {} (train MSE = {}).'
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
    # get train MSE and test MSE with different theta
    train_mse_dict, test_mse_dict = calculate_trn_test_mse(trn_file_name, test_file_name)
    # get  minimum test MSE and corresponding theta value
    theta_ = min(test_mse_dict, key=test_mse_dict.get)
    test_mse_ = test_mse_dict[theta_]
    # output
    logger.info('For dataset {} & {}, with lambda = {}, we get min test MSE = {}.'
                .format(trn_file_name, test_file_name, theta_, test_mse_))


def calculate_loss_function(x_arr, y, coefficient):
    """

    @param x_arr:
    @type x_arr:
    @param y:
    @type y:
    @param coefficient:
    @type coefficient:
    @return:
    @rtype:
    """

    pass


if __name__ == '__main__':

    # read_dataset(file_name='train-100-10')
    train_dataset = ['train-100-10', 'train-100-100', 'train-1000-100',
                     'train-50(1000)-100', 'train-100(1000)-100', 'train-150(1000)-100']
    test_dataset = ['test-100-10', 'test-100-100', 'test-1000-100',
                    'test-1000-100', 'test-1000-100', 'test-1000-100']
    for trn, test in zip(train_dataset, test_dataset):
        get_min_test_mse(trn_file_name=trn, test_file_name=test)

    # print(y_label)
    # print(train_x.dot(weight))
    # read_dataset(file_name='train-1000-100')
    # read_dataset(file_name='train-50(1000)-100')
    # read_dataset(file_name='train-100(1000)-100')
    # read_dataset(file_name='train-150(1000)-100')
