import re
import csv
import torch

def number_format_normalize(num):
    """
    From "xx,yyy,zzz.www" to "xxyyyzzz.www"
    """
    parts = num.split(',')
    result = 0
    for part in parts:
        result = result * 1000 + eval(part)
    return result


def load_nasdaq100():
    """
    Load the NASDQ-100 data.
    Only the date and closing price are counted.
    Return a 2-d tensor, where each line is a sample point in the following form:

    ...
    [year, month, day, value/label]
    ...

    """
    reader = csv.reader(open('data/dataset/nasdaq100.csv', 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.append([eval(date[0]), eval(date[1]), eval(date[2]), number_format_normalize(item[1])])
    result = torch.tensor(result)
    print('MASDQ-100 data, {} samples loaded.'.format(result.size()[0]))
    return result


def load_dax30():
    """
    Load the DAX-30 data.
    Only the date and closing price are counted.
    Return a 2-d tensor, where each line is a sample point in the following form:

    ...
    [year, month, day, value/label]
    ...

    """
    reader = csv.reader(open('data/dataset/dax30.csv', 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.append([eval(date[0]), eval(date[1]), eval(date[2]), number_format_normalize(item[1])])
    result = torch.tensor(result)
    print('DAX-30 data, {} samples loaded.'.format(result.size()[0]))
    return result


def load_shangzheng50():
    """
    Load the 上证-50 data.
    Only the date and closing price are counted.
    Return a 2-d tensor, where each line is a sample point in the following form:

    ...
    [year, month, day, value/label]
    ...

    """
    reader = csv.reader(open('data/dataset/shangzheng50.csv', 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.append([eval(date[0]), eval(date[1]), eval(date[2]), number_format_normalize(item[1])])
    result = torch.tensor(result)
    print('上证-50 data, {} samples loaded.'.format(result.size()[0]))
    return result


def time_avg_op(data, time_gap):
    """
    Average data in one period denoted by time_gap to avoid missing values.
    """
    