# %%
import networkx as nx
G = nx.Graph()

# %%
import random

num_traders = 1000

for i in range(num_traders):
    avg_profit = random.uniform(0, 1000)
    trade_frequency = random.randint(5, 20)
    G.add_node(i, avg_profit=avg_profit, trade_frequency=trade_frequency)

# %%
num_cliques = 5
clique_size = 3

for _ in range(num_cliques):
    clique = random.sample(range(num_traders), clique_size)
    for i in range(clique_size):
        for j in range(i+1, clique_size):
            G.add_edge(clique[i], clique[j], relationship='insider')

# %%
def simulate_trades(G, days=30):
    for day in range(days):
        print(f"Day {day + 1}:")
        
        # Simulate trades for each trader
        for trader in G.nodes:
            trade_activity = G.nodes[trader]['trade_frequency'] * random.uniform(0.5, 1.5)  # Randomize based on frequency

        # Track insider activities
        insider_traders = [node for node in G.nodes if any('insider' in G[node][nbr].get('relationship', '') for nbr in G.neighbors(node))]
        print(f"Insider traders' activities: {len(insider_traders)} traders are involved in suspicious activity.\n")

        
# %%
simulate_trades(G)
# %%
