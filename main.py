# -*- coding: utf-8 -*-
# @Time    : 2/4/21 5:20 PM
# @Author  : Hobey Wong
# @Contact : huangxinhong@cvte.com
# @Desc    :


from modeling import create_new_dataset


def create_new_data():
    """
        Start the experiment by creating 3 additional training les from the train-1000-100.csv
        by taking the first 50, 100, and 150 instances respectively.
        Call them: train-50(1000)-100.csv, train-100(1000)-100.csv, train-150(1000)-100.csv.
        The corresponding test file for
        these dataset would be test-1000-100.csv and no modification is needed.
    @return:
    @rtype:
    """

    for num in [50, 100, 150]:

        create_new_dataset(num_of_data=num)


if __name__ == '__main__':

    create_new_data()
