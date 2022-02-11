import re
import csv
from pandas import Period
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


MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]


def get_absolute_date(year, month, day):
    """
    For simplicity, assume all February has 29 days.
    """
    result = (year-2000) * 366
    result += MONTH_OFFSET[month-1]
    result += day
    return result


def load_data(path, name):
    """
    Load data.
    Only the date and closing price are counted.
    Return a 2-d list, where each line is a sample point in the following form:

    ...
    [absolute date, value/label]
    ...

    The absolute date is the offset from '2000-01-01'.
    """
    reader = csv.reader(open(path, 'r', encoding='utf-8'))
    result = []
    pass_first_line = True
    for item in reader:
        if pass_first_line:
            pass_first_line = False
            continue
        date = re.split('[年月日]', item[0])
        result.insert(0, [get_absolute_date(eval(date[0]), eval(date[1]), eval(date[2])), number_format_normalize(item[1])])
    print('{} data, {} samples loaded.'.format(name, len(result)))
    return result


def get_time_bucket(date, time_gap):
    return (date-1) // time_gap + 1


def data_filter(data, time_gap, time_start=None, time_end=None):
    """
    Average data in one period denoted by time_gap to avoid missing values.
    Choose only samples in the range [time_start, time_end], where both time_start
    and time_end are absolute dates. If these parameters
    are None object, no filtering applied.
    """
    n_samples = len(data)
    result = []
    avg_total, avg_num = 0, 0  # for computing the average
    lst_date = -1  # the last date cutting point
    for sample_idx in range(n_samples):
        date = data[sample_idx][0]
        value = data[sample_idx][1]
        if time_start is not None and date < time_start:
            continue
        if time_end is not None and date > time_end:
            break
        if lst_date > 0 and get_time_bucket(date, time_gap) - lst_date > 1:
            ex = Exception('The time gap is too small, please reset a proper gap.')
            raise ex
        if lst_date < 0:
            lst_date = get_time_bucket(date, time_gap)
            avg_total, avg_num = value, 1
        else:
            if get_time_bucket(date, time_gap) == lst_date:
                avg_total += value
                avg_num += 1
            else:
                result.append([lst_date, avg_total/avg_num])
                avg_total, avg_num = value, 1
                lst_date = get_time_bucket(date, time_gap)
    return result


def make_dataset(values, input_step_len, output_step_len):
    input_seq, output_seq = [], []
    n_samples = len(values)
    for start_idx in range(n_samples):
        if start_idx + input_step_len + output_step_len - 1 >= n_samples:
            break
        input_seq.append(values[start_idx:start_idx+input_step_len])
        output_seq.append(values[start_idx+input_step_len:start_idx+input_step_len+output_step_len])
    return input_seq, output_seq
