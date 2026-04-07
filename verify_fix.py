import sys
import os
import networkx as nx
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from algorithms.hpso_priority import hpso_embed

# Create substrate
sub = nx.Graph()
sub.add_node(0, cpu=10, domain=1)
sub.add_node(1, cpu=10, domain=1)
sub.add_edge(0, 1, bw=10, bw_cost=1)  # Added bw_cost

# Create VNR that is impossible (CPU > 10)
vnr = nx.Graph()
vnr.add_node(0, cpu=100, domain=1) # Impossible
vnr.add_edge(0, 0, bw=0)

print("Running hpso_embed on impossible VNR...")
try:
    result = hpso_embed(sub, vnr, particles=2, iterations=2)
    print(f"Result: {result}")

    if result is None:
        print("SUCCESS: Result is None")
    else:
        print("FAILURE: Result is not None (likely (None, None))")
except Exception as e:
    print(f"CRASHED: {e}")
