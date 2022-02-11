# 数据加载

```python
import data.load as load

nasdaq100 = load.data_filter(load.load_nasdaq100, 7)
dax30 = load.data_filter(load.load_dax30, 7)
```

两个数据集的加载（shangzheng50不使用）。

每个数据集得到一个列表，列表的每个元素仍是一个列表（包括时间桶编号、收盘价），表示一个个样本。例如：

```python
dax30[0][0]  # dax30数据集的第0号样本的时间桶编号
dax30[0][1]  # dax30数据集的第0号样本的收盘价
```

