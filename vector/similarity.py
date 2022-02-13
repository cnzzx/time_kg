import math
file_path = '偶数行.txt'
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(930):
    for j in range(930):
        A[i][j] = cos_sim(line,line)


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