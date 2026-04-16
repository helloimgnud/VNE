"""
src/scripts/train_generalize.py
===============================
Wrapper script to train PPO across varying network topologies and VNR patterns
to improve generalizability of the RL model.

It automatically finds the latest checkpoint in `checkpoints/` and resumes from it.
"""

import os
import glob
import shlex
import subprocess
import random

def get_latest_checkpoint(checkpoints_dir="checkpoints"):
    files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    if not files:
        return None
    # Sort files by modification time
    return max(files, key=os.path.getmtime)

def main():
    print("=" * 60)
    print(" RL GENERALIZATION TRAINING ".center(60))
    print("=" * 60)

    # 1. Hyper-parameters for generalization
    sub_node_configs = [40, 50, 60, 80, 100]
    vnr_node_configs = [3, 4, 6]
    vnr_batch_configs = [10, 15, 20]
    
    num_phases = 5 # Number of generalization rounds to perform
    total_steps_per_phase = 100_000

    for phase in range(1, num_phases + 1):
        print(f"\\n{'=' * 60}")
        print(f" GENERALIZATION PHASE {phase}/{num_phases} ".center(60))
        print("=" * 60)

        # Randomly select configuration for this generalization phase
        sub_nodes = random.choice(sub_node_configs)
        vnr_nodes = random.choice(vnr_node_configs)
        vnr_batch = random.choice(vnr_batch_configs)
        
        # Check for latest checkpoint (which will naturally update as phases progress)
        last_ckpt = get_latest_checkpoint()
        
        print(f"Target Substrate Nodes: {sub_nodes}")
        print(f"Target VNR Nodes Mean : {vnr_nodes}")
        print(f"Batch Size (VNRs)     : {vnr_batch}")
        if last_ckpt:
            print(f"Resuming from: {last_ckpt}")
        else:
            print("No checkpoint found. Training from scratch.")
        print("-" * 60)
        
        # Build training command
        cmd = [
            "python", "-m", "src.training.train_ppo",
            "--total-steps", str(total_steps_per_phase),
            "--sub-nodes", str(sub_nodes),
            "--vnr-nodes", str(vnr_nodes),
            "--vnr-batch", str(vnr_batch),
            "--run-name", f"ppo_generalize_phase{phase}"
        ]
        
        if last_ckpt:
            cmd.extend(["--load-checkpoint", last_ckpt])
            
        print(f"Running command: {' '.join(cmd)}\\n")
        
        # Execute
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        subprocess.run(cmd, env=env)
        
    print(f"\\n[OK] Generalization training complete across {num_phases} diverse phases!")

if __name__ == "__main__":
    main()
