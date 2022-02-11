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


def get_absolute_date(year, month, day):
    """
    For simplicity, assume all February has 29 days.
    """
    result = (year-2001) * 366
    if month-1 == 2:
        result += 29
    elif (month-1)%2 == 1 and month-1 <= 7:
        result += 31
    elif (month-1)%2 == 1 and month-1 > 7:
        result += 30
    elif (month-1)%2 == 0 and month-1 <= 7:
        result += 30
    else:  # even month larget than 7
        result += 31
    result += day
    return result


def load_nasdaq100():
    """
    Load the NASDQ-100 data.
    Only the date and closing price are counted.
    Return a 2-d tensor, where each line is a sample point in the following form:

    ...
    [absolute date, value/label]
    ...

    The absolute date is the offset from '2000-01-01'.
    """
    reader = csv.reader(open('data/dataset/nasdaq100.csv', 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.append([get_absolute_date(eval(date[0]), eval(date[1]), eval(date[2])), number_format_normalize(item[1])])
    result = torch.tensor(result)
    print('MASDQ-100 data, {} samples loaded.'.format(result.size()[0]))
    return result


def load_dax30():
    """
    Load the DAX-30 data.
    Only the date and closing price are counted.
    Return a 2-d tensor, where each line is a sample point in the following form:

    ...
    [absolute date, value/label]
    ...

    The absolute date is the offset from '2000-01-01'.
    """
    reader = csv.reader(open('data/dataset/dax30.csv', 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.append([get_absolute_date(eval(date[0]), eval(date[1]), eval(date[2])), number_format_normalize(item[1])])
    result = torch.tensor(result)
    print('DAX-30 data, {} samples loaded.'.format(result.size()[0]))
    return result


def load_shangzheng50():
    """
    Load the 上证-50 data.
    Only the date and closing price are counted.
    Return a 2-d tensor, where each line is a sample point in the following form:

    ...
    [absolute date, value/label]
    ...

    The absolute date is the offset from '2000-01-01' + 1.
    """
    reader = csv.reader(open('data/dataset/shangzheng50.csv', 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.append([get_absolute_date(eval(date[0]), eval(date[1]), eval(date[2])), number_format_normalize(item[1])])
    result = torch.tensor(result)
    print('上证-50 data, {} samples loaded.'.format(result.size()[0]))
    return result


def data_filter(data, time_gap, time_start=None, time_end=None):
    """
    Average data in one period denoted by time_gap to avoid missing values.
    Choose only samples in the range [time_start, time_end], where both time_start
    and time_end are absolute dates. If these parameters
    are None object, no filtering applied.
    """
    n_samples = data.size()[0]
    result = []
    avg_total, avg_num = 0, 0  # for computing the average
    lst_date = -1000  # the last date cutting point denoted by time_gap
    for sample_idx in range(n_samples):
        date = data[sample_idx][0]
        value = data[sample_idx][1]
        if date < time_start:
            continue
        if date > time_end:
            break
        if date-lst_date > time_gap:
            if lst_date > 0:  # at least one period started
                if avg_num == 0:
                    ex = Exception('The time gap is too small.')
                    raise ex
                result.append(torch.tensor([lst_date, avg_total/avg_num]))
            avg_total, avg_num = value, 1
            lst_date = date
        else:
            avg_total += value
            avg_num += 1
    if avg_num == 0:
        ex = Exception('The time span is too short, please reset time_start and time_end')
        raise ex
    result.append(torch.tensor([lst_date, avg_total/avg_num]))
    result = torch.tensor(result)
    return result
