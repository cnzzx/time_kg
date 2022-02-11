import data.load as load


nasdaq100 = load.load_nasdaq100()
dax30 = load.load_dax30()
shangzheng50 = load.load_shangzheng50()

f_nasdaq100 = load.data_filter(nasdaq100, 7)
print(len(f_nasdaq100))
f_dax30 = load.data_filter(dax30, 7)
print(len(f_dax30))
