import data.load as load


nasdaq100 = load.data_filter(load.load_data('data/dataset/nasdaq100.csv', 'nasdaq100'), 7)
dax30 = load.data_filter(load.load_data('data/dataset/dax30.csv', 'dax30'), 7)
shangzheng50 = load.load_data('data/dataset/shangzheng50.csv', 'shangzheng50')

print(len(nasdaq100))
print(len(dax30))
