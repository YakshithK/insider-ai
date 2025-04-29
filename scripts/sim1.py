import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

G = nx.Graph()

for i in range(20):
    G.add_node(i, name=f"Trader_{i}")

for _ in range(40):
    a, b = random.sample(range(20), 2)
    G.add_edge(a, b)

for node in G.nodes():
    trades = random.randint(10, 100)
    volume = random.uniform(1000, 100000)
    wins = random.randint(0, trades)
    win_loss = wins / trades

    G.nodes[node]["trades"] = trades
    G.nodes[node]["volume"] = volume
    G.nodes[node]["win_loss"] = win_loss

volumes = np.array([G.nodes[node]["volume"] for node in G.nodes()])
z_scores = (volumes - volumes.mean()) / volumes.std()

for i, node in enumerate(G.nodes()):
    G.nodes[node]["volume_z"] = z_scores[i]

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=800)
plt.title("Trader Network")
plt.show()

degress = [G.degree(n) for n in G.nodes()]

plt.hist(degress, bins=range(1, max(degress) + 2), align='left', color='skyblue', edgecolor='black')
plt.xlabel("Degree")
plt.ylabel("Number of Traders")
plt.title("Degree Distribution of Trader Network")
plt.show()