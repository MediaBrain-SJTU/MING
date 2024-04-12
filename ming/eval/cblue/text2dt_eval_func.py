
# 将符合诊疗决策树约束的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
import copy


def nodematrix(tree):
    nodelist=[]
    for i in range(len(tree)):
        nodelist.append(tree[i]["role"])
    node_matrix = [[0 for i in range(len(nodelist))] for j in range(len(nodelist))]
    
    # if len(tree) == 0:
    #    return (node_matrix) 
    
    count = 0
    while (nodelist[0] != 'D'):
        for i in range(len(nodelist)):
            if nodelist[i] == 'C':
                flag, leaf1, leaf2 = 0, 0, 0
                for j in range(i+1,len(nodelist)):
                    if nodelist[j]=='D' and flag==0:
                        flag = 1
                        leaf1 = j
                    elif nodelist[j]=='X' :
                        continue
                    elif nodelist[j]=='D' and flag==1:
                        #print(i)
                        leaf2 = j
                        nodelist[i]='D'
                        node_matrix[leaf1][i] = 'F'
                        node_matrix[leaf2][i] = 'F'
                        node_matrix[i][leaf1] = 'L'
                        node_matrix[i][leaf2] = 'R'
                        for k in range(i+1, leaf2+1):
                            nodelist[k]='X'
                        flag = 2
                        break
                    elif nodelist[j] == 'C':
                        break
                if flag == 2:
                    break

        count += 1
        if count > 100:
            break

    return(node_matrix)

# 计算两个节点的距离
def node_dis(node1,node2):
    if node2 is None :
        #node2 = {"role": "", "triples": [], "logical_rel": ""}
        node2 = {"role": "", "triples": [], "logical_rel": "null"}
    dis = 0
    if node1["role"] != node2["role"]:
        dis += 1
    #print(dis)
    if node1["logical_rel"] != node2["logical_rel"]:
        dis += 1
    dis += len(list((set(node1["triples"])|set(node2["triples"]))-(set(node1["triples"])&set(node2["triples"]))))
    return(dis)

# 判断两条路径是否相同
def is_path_equal(path1,path2):
    if (len(path1)!=len(path2)):
        return False
    for i in range(len(path1)):
        if isinstance(path1[i],dict) and isinstance(path2[i],dict):
            if path1[i]['role'] == path2[i]['role'] and path1[i]['logical_rel'] == path2[i]['logical_rel'] and \
                    set(path1[i]['triples']) == set(path2[i]['triples']):
                continue
            else:
                return False
        elif path1[i] != path2[i]:
            return False
    return True

# 判断两棵树是否相同
def is_tree_equal(predict_tree,gold_tree):
    if len(predict_tree) != len(gold_tree):
        return 0
    else:
        for i in range(len(predict_tree)):
            if predict_tree[i]['role'] == gold_tree[i]['role'] and \
                    predict_tree[i]['logical_rel'] == gold_tree[i]['logical_rel'] and \
                    set(predict_tree[i]['triples']) == set(gold_tree[i]['triples']):
                continue
            else:
                return 0
    return 1

# 计算模型预测的诊疗决策树和ground turth的距离，距离越小表示两树越相似，为计算编辑比率做准备
def edit_distance(predict_tree, gold_tree, predict_matrix, gold_matrix):
    dis = 0
    stack1 = [0]
    stack2 = [0]

    try:
        while stack1:
            s1=stack1.pop()
            s2=stack2.pop()
            if ('L' not in predict_matrix[s1] and 'R' not in predict_matrix[s1]) \
                    and ('L' in gold_matrix[s2] or 'R' in gold_matrix[s2]):
                dis += node_dis(predict_tree[s1], gold_tree[s2])
                stack_tmp=[]
                stack_tmp.append(gold_matrix[s2].index('R'))
                stack_tmp.append(gold_matrix[s2].index('L'))
                while stack_tmp:
                    s_tmp=stack_tmp.pop()
                    dis += node_dis(gold_tree[s_tmp],None)
                    if ('L' in gold_matrix[s_tmp] and 'R' in gold_matrix[s_tmp]):
                        stack_tmp.append(gold_matrix[s_tmp].index('R'))
                        stack_tmp.append(gold_matrix[s_tmp].index('L'))
            elif ('L' in predict_matrix[s1] and 'R' in predict_matrix[s1]) \
                    and ('L' not in gold_matrix[s2] or 'R' not in gold_matrix[s2]):
                dis += node_dis(predict_tree[s1], gold_tree[s2])
                stack_tmp=[]
                stack_tmp.append(predict_matrix[s1].index('R'))
                stack_tmp.append(predict_matrix[s1].index('L'))
                while stack_tmp:
                    s_tmp=stack_tmp.pop()
                    dis += node_dis(predict_tree[s_tmp], None)
                    if ('L' in predict_matrix[s_tmp] and 'R' in predict_matrix[s_tmp]):
                        stack_tmp.append(predict_matrix[s_tmp].index('R'))
                        stack_tmp.append(predict_matrix[s_tmp].index('L'))
            elif ('L' not in predict_matrix[s1] and 'R' not in predict_matrix[s1]) and \
                    ('L' not in gold_matrix[s2] and 'R' not in gold_matrix[s2]):
                dis += node_dis(predict_tree[s1], gold_tree[s2])
            else:
                stack1.append(predict_matrix[s1].index('R'))
                stack1.append(predict_matrix[s1].index('L'))
                stack2.append(gold_matrix[s2].index('R'))
                stack2.append(gold_matrix[s2].index('L'))
                dis += node_dis(predict_tree[s1], gold_tree[s2])

    except Exception as e:
        print("calculating edit dist wrong!")
        print(e)

    return dis

# 计算决策路径抽取的TP,TP+FP,TP+FN
def decision_path(predict_tree, gold_tree, predict_matrix, gold_matrix):
    leaf1, leaf2, paths1, paths2 = [], [], [], []

    try:
        for i in range(len(predict_matrix)):
            if ('L' not in predict_matrix[i] and 'R' not in predict_matrix[i]):
                leaf1.append(i)
        for node in leaf1:
            path=[predict_tree[node]]
            while node !=0:
                #print(predict_matrix)
                #print(node)
                #print(predict_matrix[node])
                path.append(predict_matrix[predict_matrix[node].index('F')][node])
                path.append(predict_tree[predict_matrix[node].index('F')])
                node =predict_matrix[node].index('F')
            paths1.append(path)
        for i in range(len(gold_matrix)):
            if ('L' not in gold_matrix[i] and 'R' not in gold_matrix[i]):
                leaf2.append(i)
        for node in leaf2:
            path=[gold_tree[node]]
            while node != 0:
                path.append(gold_matrix[gold_matrix[node].index('F')][node])
                path.append(gold_tree[gold_matrix[node].index('F')])
                node =gold_matrix[node].index('F')
            paths2.append(path)
        res = 0
        for path1 in paths1:
            for path2 in paths2:
                if is_path_equal(path1, path2):
                    res += 1
                    break
    except Exception as e:
        print("calculating decision path wrong!")
        print(e)
        res = 0

    return res,len(paths1),len(paths2)


# 计算三元组抽取的TP,TP+FP,TP+FN
def triplet_extraction(predict_tree, gold_tree):
    predict_triplet, gold_triplet = [], []
    for i in range(len(predict_tree)):
        for triplet in predict_tree[i]["triples"]:
            predict_triplet.append(triplet)
    for i in range(len(gold_tree)):
        for triplet in gold_tree[i]["triples"]:
            gold_triplet.append(triplet)
    predict_triplet_num = len(list(set(predict_triplet)))
    gold_triplet_num = len(list(set(gold_triplet)))
    correct_triplet_num =len(list(set(gold_triplet)&set(predict_triplet)))
    return [correct_triplet_num, predict_triplet_num, gold_triplet_num]

# 计算节点抽取的TP,TP+FP,TP+FN
def node_extraction(predict_tree, gold_tree):
    predict_node, gold_node = [], []
    for i in range(len(predict_tree)):
        if len(predict_tree[i]['triples'])>0:
            predict_node.append(predict_tree[i])
    for i in range(len(gold_tree)):
        if len(gold_tree[i]['triples']) > 0:
            gold_node.append(gold_tree[i])

    predict_triplet_num = len(predict_node)
    gold_triplet_num = len(gold_node)
    correct_triplet_num = 0
    for node1 in predict_node:
        for node2 in gold_node:
            if len(node1['triples'])>0 and node1['role'] == node2['role'] and node1['logical_rel'] == node2['logical_rel'] and set(node1['triples']) == set(node2['triples']):
                correct_triplet_num +=1
    return [correct_triplet_num, predict_triplet_num, gold_triplet_num]

#评测函数，共计算5个指标: 三元组抽取的F1；节点抽取的F1；决策树的Acc；决策路径的F1; 树的编辑距离
def text2dt_eval_single_tree(predict_tree, gold_tree):
    # 将符合诊疗决策树的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
    for node in predict_tree:
        for i in range(len(node['triples'])):
            print(node['triples'][i])
            assert len(node['triples'][i]) == 3, "the triple format is wrong"
            node['triples'][i]=(node['triples'][i][0].lower(), node['triples'][i][1].lower(), node['triples'][i][2].lower())
    for node in gold_tree:
        for i in range(len(node['triples'])):
            assert len(node['triples'][i]) == 3, "the triple format is wrong"
            node['triples'][i]=(node['triples'][i][0].lower(), node['triples'][i][1].lower(), node['triples'][i][2].lower())

    # print("step1: ")
    predict_matrix = nodematrix(predict_tree)
    gold_matrix = nodematrix(gold_tree)

    # 用于计算生成树的Acc
    tree_num = (0 if predict_tree == [] else 1)
    correct_tree_num = is_tree_equal(predict_tree,gold_tree)

    # 用于计算triplet抽取的F1
    correct_triplet_num, predict_triplet_num, gold_triplet_num = triplet_extraction(predict_tree, gold_tree)

    # 用于计算决策路径的F1
    # print("step2: ")
    correct_path_num, predict_path_num, gold_path_num = decision_path(
        copy.deepcopy(predict_tree),
        copy.deepcopy(gold_tree),
        copy.deepcopy(predict_matrix),
        copy.deepcopy(gold_matrix)
    )
    # print("correct_path_num: ", correct_path_num)

    # 用于计算树的编辑距离
    edit_dis = edit_distance(predict_tree, gold_tree, predict_matrix, gold_matrix)

    correct_node_num, predict_node_num, gold_node_num = node_extraction(predict_tree, gold_tree)

    return tree_num,correct_tree_num, correct_triplet_num, predict_triplet_num, gold_triplet_num, correct_path_num, predict_path_num, gold_path_num, edit_dis, correct_node_num, predict_node_num, gold_node_num