import networkx as nx
import  node2vec

c_s = 0
interaction_file = './yeast/4932.protein.physical.links.v11.5.txt'
G = nx.DiGraph()
with open(interaction_file, 'r') as file:
    for line_number, line in enumerate(file):
        if line_number == 0:
            continue
        parts = line.strip().split()
        if len(parts) == 3:
            node1, node2, edge_feature = parts
            if float(edge_feature) >= c_s:
                G.add_node(node1)
                G.add_node(node2)
                G.add_edge(node1, node2, weight=1.0)

edge_count = G.number_of_edges()
print(f"图的边数: {edge_count}")

model = node2vec.Node2vec(G,path_length=64,num_paths=32,p=1,q=1)

model.train(dim=100,workers=8,window_size=10)
model.save_embeddings('./graph.emb.npz')

