# -*- coding: utf-8 -*-
# @Time    : 2/4/21 5:08 PM
# @Author  : Hobey Wong
# @Contact : huangxinhong@cvte.com
# @Desc    :

import pandas as pd


def create_new_dataset(num_of_data):
    """
    Start the experiment by creating 3 additional training les from the train-1000-100.csv
    by taking the first 50, 100, and 150 instances respectively. Call them: train-50(1000)-
    100.csv, train-100(1000)-100.csv, train-150(1000)-100.csv. The corresponding test le for
    these dataset would be test-1000-100.csv and no modification is needed.
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


if __name__ == '__main__':

    pass
