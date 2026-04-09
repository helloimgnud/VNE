# scripts/test_ac_controller.py
"""
Smoke test for the RL Admission Controller.
Creates a small random substrate + VNR and verifies that
feature extraction + forward pass work without errors.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networkx as nx
import numpy as np

def make_substrate(n=20, p=0.3, seed=42):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for n_id in G.nodes():
        G.nodes[n_id]['cpu'] = np.random.randint(50, 300)
    for u, v in G.edges():
        G.edges[u, v]['bw'] = np.random.randint(500, 3000)
    return G

def make_vnr(n=4, p=0.6, seed=7):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for n_id in G.nodes():
        G.nodes[n_id]['cpu'] = np.random.randint(10, 100)
    for u, v in G.edges():
        G.edges[u, v]['bw'] = np.random.randint(100, 1000)
    return G


def main():
    from src.algorithms.ac_controller import (
        extract_features, obs_to_tensor, ACActorCritic, AdmissionController
    )
    import torch

    sub = make_substrate()
    vnr = make_vnr()
    print(f"Substrate: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
    print(f"VNR:       {vnr.number_of_nodes()} nodes, {vnr.number_of_edges()} edges")

    # 1. Feature extraction
    obs = extract_features(sub, vnr)
    print(f"\np_net_x shape:          {obs['p_net_x'].shape}")
    print(f"p_net_edge_index shape: {obs['p_net_edge_index'].shape}")
    print(f"p_net_edge_attr shape:  {obs['p_net_edge_attr'].shape}")
    print(f"v_net_x shape:          {obs['v_net_x'].shape}")
    print(f"v_net_edge_index shape: {obs['v_net_edge_index'].shape}")
    print(f"v_net_edge_attr shape:  {obs['v_net_edge_attr'].shape}")
    assert obs['p_net_x'].shape == (sub.number_of_nodes(), 4), "p_net_x wrong shape"
    assert obs['v_net_x'].shape == (vnr.number_of_nodes(), 5), "v_net_x wrong shape"
    print("[OK] Feature extraction OK")

    # 2. Forward pass (random weights)
    device = torch.device('cpu')
    t_obs = obs_to_tensor(obs, device)
    model = ACActorCritic(emb_dim=64)
    model.eval()
    with torch.no_grad():
        logits = model.act(t_obs)
    assert logits.shape == (1, 2), f"Expected (1,2), got {logits.shape}"
    print(f"[OK] Forward pass OK - logits = {logits.numpy()}")

    # 3. AdmissionController API (no pretrained model -> random weights)
    ac = AdmissionController(model_path=None, embedding_dim=64)
    s = ac.score(sub, vnr)
    a = ac.should_accept(sub, vnr)
    print(f"[OK] score  = {s:.4f}")
    print(f"[OK] accept = {a}")
    assert isinstance(s, float) and 0.0 <= s <= 1.0
    assert isinstance(a, bool)

    print("\n=== ALL TESTS PASSED ===")


if __name__ == '__main__':
    main()
