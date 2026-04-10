"""
src/training
============
Training scripts for the GCN-RL VNR Ordering Scheduler.

Modules
-------
generate_data   : substrate / VNR generation factories (wraps existing generators)
train_reinforce : Phase 1 — REINFORCE baseline training loop
train_ppo       : Phase 2 — PPO training loop (custom, CleanRL-style)
"""
