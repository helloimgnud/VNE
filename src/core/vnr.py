# src/core/vnr.py
import networkx as nx
from generators.generator import generate_vnr

class VNR:
    def __init__(self, G, lifetime=50, arrival_time=0, id=None):
        self.G = G
        self.lifetime = lifetime
        self.arrival_time = arrival_time
        self.id = id

    @classmethod
    def from_params(cls, **kwargs):
        g = generate_vnr(**kwargs)
        return cls(g)
