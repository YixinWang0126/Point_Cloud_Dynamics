import numpy as np

def persist_entropy(pd_point):
    pe = 0
    duration = pd_point[:,1] - pd_point[:,0]
    Len = duration.sum()
    for i in duration:
        l_over_L = i/Len
        value = l_over_L * np.log(l_over_L)
        pe -= value
    return pe

def hyper_graph_entropy(coot_distance):
    num_points = len(coot_distance[:,-1])
    hge = 0
    for i in range(num_points):#compute by row
        point_cycle_distance = coot_distance[i]
        Sum = point_cycle_distance.sum()
        if Sum == 0:
            continue
        for j in point_cycle_distance:
            l_over_L = j/Sum
            if l_over_L != 0:
                value = l_over_L * np.log(l_over_L)
            else:
                value = 0
            hge -= value
    hge /= num_points
    hge /= np.log(len(coot_distance[-1,:]))
    return hge

def hyper_edge_entropy(coot_distance):
    """
    计算超边视角的熵：
    coot_distance: ndarray, shape (n_points, m_edges)
        第 i 行是顶点 v_i 到每条环/超边的距离或权重。
    返回值：HE_E，标量
    """
    # 点数和超边数
    n_points, m_edges = coot_distance.shape
    hee = 0.0

    # 对每条超边（列）计算条件熵
    for j in range(m_edges):
        # 第 j 条超边对应的所有点的距离/权重
        edge_weights = coot_distance[:, j]
        total = edge_weights.sum()
        if total == 0:
            continue  # 如果这一条超边没有任何点关联，则跳过

        # 累加这一条超边的熵
        for w in edge_weights:
            p = w / total
            if p > 0:
                hee -= p * np.log(p)

    # 对所有超边取平均
    hee /= m_edges
    hee /= np.log(n_points)
    return hee
        
def hypergraph_shannon_entropy(y1: np.ndarray) -> float:
    # 构造拉普拉斯矩阵
    L = y1 @ y1.T
    # 计算特征值
    eigvals = np.linalg.eigvalsh(L)
    # 归一化
    mu = eigvals / eigvals.sum()
    mu = mu[mu > 0]
    # 计算熵
    return -np.sum(mu * np.log2(mu))