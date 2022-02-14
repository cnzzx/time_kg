import math
import numpy as np
import torch


def event_source_pro():
    file_path = 'source.txt'
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_1 = open("data/event/title.txt", 'w', encoding='utf-8')  # 奇数行
    data_2 = open("data/event/embedding.txt", 'w', encoding='utf-8')  # 偶数行
    
    num = 0  # 行数-1
    for line in lines:
        if (num % 2) == 0:  # num为偶数说明是奇数行
            print(line.strip(), file=data_1)  # .strip用来删除空行
        else:  # # num为奇数说明是偶数行
            print(line.strip(), file=data_2)
        num += 1
    
    data_1.close()
    data_2.close()


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


def event_graph_pro():
    # The size of bert embedding is 768.
    A = np.zeros((930, 768), dtype=float)  # 嵌入矩阵
    B_sim= np.zeros((930, 930), dtype=float)  # 相似度矩阵
    C_adj= np.zeros((930, 930), dtype=int)  # 邻接矩阵

    f = open('data/event/embedding.txt')  # 打开数据文件文件
    lines = f.readlines()  # 把全部数据文件读到一个列表lines中
    A_row = 0  # 表示矩阵的行，从0行开始
    for line in lines:  # 把lines中的数据逐行读取出来
        line = line.strip("\n")
        line = line.strip("\t")
        list = line.split(" ")
        A[A_row:] = list[0:768]  # 把处理后的数据放到方阵A中。
        A_row += 1  # 然后方阵A的下一行接着读

    for i in range(930):
        for j in range(i,930):
            B_sim[i][j] = cos_sim(A[i],A[j])
            print('({},{})'.format(i,j))
            if (i!=j):
                B_sim[j][i] = B_sim[i][j]
    
    np.savetxt('data/event/B_sim.csv', B_sim, delimiter = ',')
    num_1=0 # 边的数量
    for i in range(930):
        for j in range(930):
            if B_sim[i][j]>0.9:
                C_adj[i][j]=1
                num_1+=1
    np.savetxt('data/event/C_adj.csv', C_adj, delimiter = ',')
    print('The number of event edges: {}. '.format((num_1-930)/2))


MONTH_OFFSET = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]


def get_absolute_date(year, month, day):
    """
    For simplicity, assume all February has 29 days.
    """
    result = (year-2000) * 366
    result += MONTH_OFFSET[month-1]
    result += day
    return result


def get_time_bucket(date, time_gap):
    return (date-1) // time_gap + 1


def get_event_info(time_gap):
    with open('data/event/embedding.txt', 'r', encoding='utf-8') as f:
        eb_lines = f.readlines()
    with open('data/event/title.txt', 'r', encoding='utf-8') as f:
        ti_lines = f.readlines()
    n_lines = len(eb_lines)
    embeddings = torch.zeros((n_lines, 768))
    dates = torch.zeros((n_lines), dtype=torch.int32)
    for line_idx in range(n_lines):
        eb_line = eb_lines[line_idx]
        eb_line = eb_line.strip('\n')
        eb_line = eb_line.strip("\t")
        eb_list = eb_line.split(" ")
        for eb_idx in range(768):
            eb_list[eb_idx] = eval(eb_list[eb_idx])
        embeddings[line_idx] = torch.tensor(eb_list)

        ti_line = ti_lines[line_idx]
        ti_line = ti_line.strip('\n')
        ti_line = ti_line.strip('\t')
        date = ti_line.split(' ')[0]
        date_ls = date.split('-')
        for date_idx in range(3):
            date_ls[date_idx] = int(date_ls[date_idx])
        dates[line_idx] = get_time_bucket(get_absolute_date(*(date_ls)), time_gap)
    
    # sort - dates in ascending order
    n_events = len(dates)
    p_idx = torch.arange(0, n_events, 1)
    for i in range(n_events-1):
        for j in range(n_events-i-1):
            if dates[p_idx[j]] > dates[p_idx[j+1]]:
                t = p_idx[j].clone()
                p_idx[j] = p_idx[j+1].clone()
                p_idx[j+1] = t.clone()
    graph_adj = torch.tensor(np.loadtxt('data/event/C_adj.csv', delimiter=','))  # The graph sturcture.
    graph_sim = torch.tensor(np.loadtxt('data/event/B_sim.csv', delimiter=','))  # The edge features.

    sorted_embeddings = torch.zeros(embeddings.size())
    sorted_dates = torch.zeros(dates.size(), dtype=torch.int32)
    sorted_graph_adj = torch.zeros(graph_adj.size(), dtype=torch.int32)
    sorted_graph_sim = torch.zeros(graph_sim.size())

    for i in range(n_events):
        sorted_embeddings[i] = embeddings[p_idx[i]]
        sorted_dates[i] = dates[p_idx[i]]
        for j in range(n_events):
            sorted_graph_adj[i][j] = graph_adj[p_idx[i], p_idx[j]]
            sorted_graph_sim[i][j] = graph_sim[p_idx[i], p_idx[j]]

    return sorted_dates, sorted_embeddings, sorted_graph_adj, sorted_graph_sim


if __name__== '__main__':
    # event_source_pro()
    event_graph_pro()
    # dates, embeddings, graph_adj, graph_sim = get_event_info()
    # print(graph_adj.size(), graph_sim.size())
