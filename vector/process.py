# file_path = '123.txt'
# with open(file_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()
# data_1 = open("奇数行.txt", 'w', encoding='utf-8')
# data_2 = open("偶数行.txt", 'w', encoding='utf-8')
#
# num = 0  # 行数-1
# for line in lines:
#     if (num % 2) == 0:  # num为偶数说明是奇数行
#         print(line.strip(), file=data_1)  # .strip用来删除空行
#     else:  # # num为奇数说明是偶数行
#         print(line.strip(), file=data_2)
#     num += 1
#
# data_1.close()
# data_2.close()

from numpy import *
import math
import numpy

A = zeros((930, 768), dtype=float)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
B_sim= zeros((930, 930), dtype=float)
C_adj= zeros((930, 930), dtype=int)

f = open('偶数行.txt')  # 打开数据文件文件
lines = f.readlines()  # 把全部数据文件读到一个列表lines中
A_row = 0  # 表示矩阵的行，从0行开始
for line in lines:  # 把lines中的数据逐行读取出来

    line = line.strip("\n")
    line = line.strip("\t")
    list = line.split(" ")
    # line = [float(x) for x in line]
    # list =line.strip('\n').split(' ')# 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
    A[A_row:] = list[0:768]  # 把处理后的数据放到方阵A中。
    A_row += 1  # 然后方阵A的下一行接着读
    # print(line)

print(A)  # 打印 方阵A里的数据
def cos_sim(s1_cut_code, s2_cut_code):
    # 计算余弦相似度
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(s1_cut_code)):
        sum += s1_cut_code[i] * s2_cut_code[i]
        sq1 += pow(s1_cut_code[i], 2)
        sq2 += pow(s2_cut_code[i], 2)

    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 3)
    except ZeroDivisionError:
        result = 0.0
    #     print("余弦相似度为：%f"%result)
    return result

for i in range(930):
    for j in range(i,930):
        B_sim[i][j] = cos_sim(A[i],A[j])
        if (i!=j):
            B_sim[j][i] = B_sim[i][j]

print (B_sim)
numpy.savetxt('B_sim.csv', B_sim, delimiter = ',')
num_1=0 #边的数量
for i in range(930):
    for j in range(930):
        if B_sim[i][j]>0.85:
            C_adj[i][j]=1
            num_1+=1
print(C_adj)
numpy.savetxt('C_adj.csv', C_adj, delimiter = ',')
print((num_1-930)/2)
