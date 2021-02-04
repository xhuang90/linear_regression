# -*- coding: utf-8 -*-
# @Time    : 2/4/21 5:34 PM
# @Author  : Hobey Wong
# @Contact : huangxinhong@cvte.com
# @Desc    :

"""
Implement L2 regularized linear regression algorithm with  ranging from 0 to 150
(integers only). For each of the 6 dataset, plot both the training set MSE and the test
set MSE as a function of  (x-axis) in one graph.
"""

import pandas as pd
import numpy as np


def read_dataset(file_name):
    """

    @param file_name:
    @type file_name:
    @return:
    @rtype:
    """
    # TODO read_dataset(file_name='train-100-10') NAN col
    df_dataset = pd.read_csv('data/{}.csv'.format(file_name))
    df_dataset = df_dataset.dropna(how='all', axis=1)
    print(df_dataset.head())
    df_x_val = df_dataset.drop(columns='y', axis=1)
    df_t_val = df_dataset['y']
    print(file_name)
    print(df_x_val.head())
    print(df_t_val)


if __name__ == '__main__':

    # read_dataset(file_name='train-100-10')
    read_dataset(file_name='train-100-100')
    # read_dataset(file_name='train-1000-100')
    # read_dataset(file_name='train-50(1000)-100')
    # read_dataset(file_name='train-100(1000)-100')
    # read_dataset(file_name='train-150(1000)-100')
