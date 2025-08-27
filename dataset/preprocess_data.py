import json
from collections import defaultdict  # 使用 collections.defaultdict 而不是 typing.DefaultDict

with open("../dataset/Laptops_corenlp/train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head'])
        max_len = len(head)  # 避免使用 `max` 作为变量名（会覆盖内置函数）

        # 构建邻接矩阵
        tmp = [[0] * max_len for _ in range(max_len)]
        for i in range(max_len):
            # 获取第 i 和 j 个词的关系
            j = head[i]
            if j == 0:
                continue
            # j - 1 是因为索引从 0 开始
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        # 使用 collections.defaultdict 创建默认字典
        tmp_dict = defaultdict(list)  # 这里直接初始化 defaultdict，默认值是空列表

        # 得到 tmp_dict 是一个邻接表，表示每个词的直接邻居
        for i in range(max_len):
            for j in range(max_len):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        # 表示词 i 到词 j 的层次距离（基于句法树的路径长度）
        level_degree = [[5] * max_len for _ in range(max_len)]

        for i in range(max_len):
            node_set = set()
            # 初始化对角线，表示到自己的距离是 0
            level_degree[i][i] = 0
            node_set.add(i)

            # 遍历 i 的直接邻居，i -> j
            for j in tmp_dict[i]:
                # 如果未被访问
                if j not in node_set:
                    # i 到 j 的距离为1
                    level_degree[i][j] = 1
                    node_set.add(j)
                # 遍历每个邻居 j 的邻居 k，i -> k -> j
                for k in tmp_dict[j]:
                    if k not in node_set:
                        # i 到 k 的距离为2
                        level_degree[i][k] = 2
                        node_set.add(k)
                        # 遍历每个邻居 k 的邻居 g，i -> k -> j -> g
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                level_degree[i][g] = 3
                                node_set.add(g)
                                # 遍历每个邻居 g 的邻居 q，i -> k -> j -> g -> q
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                        level_degree[i][q] = 4
                                        node_set.add(q)
        d['short'] = level_degree


    wf = open('../dataset/Laptops_corenlp/train_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Laptops_corenlp/test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    for d in data:
        head = list(d['head'])
        max=len(head)
        tmp = [[0]*max for _ in range(max)]
        for i in range(max):
            j=head[i]
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1

        tmp_dict = defaultdict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g)
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q)
        d['short'] = leverl_degree


    wf = open('../dataset/Laptops_corenlp/test_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Restaurants_corenlp/train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head'])
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]
        for i in range(max):
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1

        tmp_dict = defaultdict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g)
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q)
        d['short'] = leverl_degree


    wf = open('../dataset/Restaurants_corenlp/train_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Restaurants_corenlp/test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head'])
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]
        for i in range(max):
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1

        tmp_dict = defaultdict(list)

        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g)
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q)
        d['short'] = leverl_degree


    wf = open('../dataset/Restaurants_corenlp/test_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

with open("../dataset/Tweets_corenlp/train.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head']) 
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = defaultdict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Tweets_corenlp/train_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close() 



with open("../dataset/Tweets_corenlp/test.json", 'r') as f:
    all_data = []
    data = json.load(f)
    # print(data)
    for d in data:
        # print(d)
        # exit()
        head = list(d['head']) 
        max=len(head)
        # print(head)
        # print(max)
        tmp = [[0]*max for _ in range(max)]  
        for i in range(max): 
            j=head[i]
            #print(j)
            if j==0:
                continue
            tmp[i][j-1]=1
            tmp[j-1][i]=1
        
        tmp_dict = defaultdict(list)
        
        for i in range(max):
            for j in range(max):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)  

        leverl_degree = [[5]*max for _ in range(max)]

        for i in range(max):
            node_set = set()
            leverl_degree[i][i]=0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    leverl_degree[i][j]=1
                    #print(word_leverl_degree)
                    node_set.add(j)
                for k in tmp_dict[j]:
                    #print(tmp_dict[j])
                    if k not in node_set:
                        leverl_degree[i][k] = 2
                        #print(word_leverl_degree)
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                leverl_degree[i][g] = 3
                                #print(word_leverl_degree)
                                node_set.add(g) 
                                for q in tmp_dict[g]:
                                    if q not in node_set:
                                       leverl_degree[i][q] = 4
                                       #print(word_leverl_degree)
                                       node_set.add(q) 
        d['short'] = leverl_degree


    wf = open('../dataset/Tweets_corenlp/test_write.json', 'w')
    wf.write(json.dumps(data, indent=4))
    wf.close()

     