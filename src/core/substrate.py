# src/core/substrate.py
import networkx as nx
from generators.generator import generate_substrate
from src.utils.graph_utils import copy_substrate, cpu_free_list

class Substrate:
    def __init__(self, G):
        self.G_original = G
        self.reset()

    @classmethod
    def from_params(cls, **kwargs):
        g = generate_substrate(**kwargs)
        return cls(g)

    def reset(self):
        self.G = copy_substrate(self.G_original)

    def snapshot_resources(self):
        return cpu_free_list(self.G)
