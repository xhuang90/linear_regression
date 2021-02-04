# -*- coding: utf-8 -*-
# @Time    : 2/4/21 5:20 PM
# @Author  : Hobey Wong
# @Contact : huangxinhong@cvte.com
# @Desc    :


from data_preprocess import create_new_dataset


def create_new_data():

    for num in [50, 100, 150]:
        create_new_dataset(num_of_data=num)


if __name__ == '__main__':

    create_new_data()
