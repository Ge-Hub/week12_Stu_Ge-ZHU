"""
Dolphin数据集是 D.Lusseau 等人使用长达 7 年的时间观察新西兰 Doubtful Sound海峡 62 只海豚群体的交流情况而得到的海豚社会关系网络。这个网络具有 62 个节点，159 条边。节点表示海豚，边表示海豚间的频繁接触
数据文件：dolphins.gml
1）对Dolphin 关系进行Graph Embedding，使用GCN
2）对Embedding进行可视化（使用PCA呈现在二维平面上）
"""

# 导入库实现GCN
import networkx as nx
from networkx import to_numpy_matrix
import matplotlib.pyplot as plt
import numpy as np

# 数据加载，构造图
G = nx.read_gml('dolphins.gml')

# 对网络G进行可视化
def plot_graph(G):
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    nx.draw_networkx(G, pos);
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=300, node_color='r', alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha =0.4)
    plt.show()



# 可视化
print(G)
print(type(G))
plot_graph(G)
#print(list(G.nodes()))
#G.nodes()

# 构建GCN，计算A_hat和D_hat矩阵
order = sorted(list(G.nodes()))
A = to_numpy_matrix(G, nodelist=order)
print('A=\n', A)

# 生成对角矩阵
I = np.eye(G.number_of_nodes())
A_hat = A + I
print('A_hat=\n', A_hat)

# 得到对角线上的元素
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))
print('D_hat=\n', D_hat)

# 初始化权重, normal 正态分布 loc均值 scale标准差
W_1 = np.random.normal(loc=0, scale=1, size=(G.number_of_nodes(), 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

# 激活函数ReLU
# x<0时 结果=0; x>=0时，结果=x
def relu(x):
    return(abs(x)+x)/2

# 叠加GCN层，这里只使用单位矩阵作为特征表征，即每个节点被表示为一个 one-hot 编码的类别变量
def gcn_layer(A_hat, D_hat, X, W):
	return relu(D_hat**-1 * A_hat * X * W)
H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2
print('output=\n', output)

# 提取特征表征
feature_representations = {}
nodes = list(G.nodes())
for i in range(len(nodes)):
    feature_representations[nodes[i]] = np.array(output)[i]
print('feature_representations=\n', feature_representations)

# 不同节点value，绘制不同的颜色
def getValue(value):
    colorList = ['blue','green','purple','yellow','red','pink','orange','black','white','gray','brown','wheat']
    return colorList[int(value)]

# 绘制所选节点的向量
def plot_nodes(nodes, output):
    # 绘制节点向量
    plt.figure(figsize=(12,9))
    # 创建一个散点图的投影
    plt.scatter(output[:, 0], output[:, 1])
    for i, node in enumerate(nodes):
        plt.annotate(node, xy=(output[i, 0], output[i, 1]))        
    plt.show()

plot_nodes(nodes, np.array(output))

# 去除激活函数ReLU, 比较效果
def gcn_layer(A_hat, D_hat, X, W):
    return D_hat**-1 * A_hat * X * W
H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

plot_nodes(nodes, np.array(output))

