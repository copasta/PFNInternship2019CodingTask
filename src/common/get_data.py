import numpy as np

def get_train(f_size=2000):

    # datasetsに含まれるtrainデータの取得

    train_node_size = []
    train_H = []
    train_y = []

    # adjacency matrix
    for i in range(f_size):
        with open("src/datasets/train/{}_graph.txt".format(i)) as f:
            train_node_size.append(list(map(int,f.readline().split())))
            data = []
            for line in f.readlines():
                data.append(list(map(int, line.split())))
            train_H.append(np.array(data))
    train_H = np.array(train_H)
    train_node_size = np.array(train_node_size)

    # label
    for i in range(f_size):
        with open("src/datasets/train/{}_label.txt".format(i)) as f:
            train_y.append(np.array(list(map(int,f.readline().split()))))
    train_y = np.array(train_y)

    return train_H, train_y, train_node_size

def get_test(f_size=500):
    
    # datasetsに含まれるtestデータの取得

    test_node_size = []
    test_H = []

    # adjacency matrix
    for i in range(f_size):
        with open("src/datasets/test/{}_graph.txt".format(i)) as f:
            test_node_size.append(list(map(int,f.readline().split())))
            data = []
            for line in f.readlines():
                data.append(list(map(int, line.split()))) 
            test_H.append(np.array(data))
    test_H = np.array(test_H)
    test_node_size = np.array(test_node_size)

    return test_H, test_node_size

def shuffle_split(train_H, train_y, graph_size, split_size=0.7, seed=1234):
    
    # trainデータをtrainとvalidationに分割
    
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(train_H.shape[0])
    temp_H = train_H[shuffle_idx]
    temp_y = train_y[shuffle_idx]
    temp_graph = graph_size[shuffle_idx]

    split_idx = int(len(temp_H) * split_size)

    train_H, val_H = temp_H[:split_idx], temp_H[split_idx:]
    train_y, val_y = temp_y[:split_idx], temp_y[split_idx:]
    train_graph_size, val_graph_size = temp_graph[:split_idx], temp_graph[split_idx:]

    return train_H, train_y, val_H, val_y, train_graph_size, val_graph_size

def get_feature(D, H, node_size):

    # 特徴ベクトルの生成
    # 各ノードに連結しているノードの数を特徴として捉えたベクトルを利用した．

    x = []
    for idx in range(len(node_size)):
        graph = node_size[idx][0]
        temp = np.zeros((graph, D))
        for num in range(graph):
            temp[num][np.sum(H[idx][num])%D] = np.sum(H[idx][num])
        x.append(temp)
    x = np.array(x)
    for j in range(len(x)):
        x[j] = x[j].T
    return x