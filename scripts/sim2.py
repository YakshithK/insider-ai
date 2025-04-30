import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from torch_geometric.utils import from_networkx
from torch_geometric_temporal.nn.recurrent import TGCN
import torch.nn.functional as F
import torch
from torch.nn import Linear

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

for u, v in G.edges():
    G[u][v]['day'] = random.randint(0, 6)

snapshots = []

for day in range(7):
    edges_tdy = [(u, v) for u, v, d in G.edges(data="day") if d == day]
    H = G.edge_subgraph(edges_tdy).copy()
    snapshots.append(H)

pyg_snapshots = []

for H in snapshots:
    for n in H.nodes():
        H.nodes[n]['x'] = [
            H.nodes[n]['volume_z'],
            H.nodes[n]['win_loss']
        ]
    data = from_networkx(H)
    pyg_snapshots.append(data)

class myTCGN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tcgn = TGCN(in_channels, out_channels)
        self.lin = Linear(out_channels, 1)
    def forward(self, snapshot, h):
        h_new = self.tcgn(snapshot.x, snapshot.edge_index, h)
        out = torch.sigmoid(self.lin(h_new))
        return out, h_new

model = myTCGN(2, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
labels = [torch.tensor([H.nodes[n].get('insider', 0) for n in H.nodes()], dtype=torch.float) for H in snapshots]

h = None
for epoch in range(20):
    model.train()
    h = None
    total_loss = 0.0
    for data, y in zip(pyg_snapshots, labels):
        optimizer.zero_grad()
        out, h = model(data, h)
        loss = F.binary_cross_entropy(out.view(-1), y)
        loss.backward(); optimizer.step();
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}, Loss: {total_loss:.4f}")

model.eval()
h = None
preds, trues = [], []

with torch.no_grad():
    for data, y in zip(pyg_snapshots, labels):
        out, h = model(data, h)
        preds.append(out.view(-1) > 0.5).float()
        trues.append(y)

preds = torch.cat(preds)
trues = torch.cat(trues)

accuracy= (preds == true).float().mean().item()
print(f"Accuracy: {accuracy:.4f}")