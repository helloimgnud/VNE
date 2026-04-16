# src/generators_v2/dataset_generator.py
"""
Dataset Generator — v2

Two orchestration classes:

1. DatasetGeneratorV2
   ─ Extends the original DatasetGenerator from src/generators/dataset_generator.py.
   ─ Uses generators_v2 substrate/VNR generators for all existing experiment types.
   ─ Adds three new experiment presets:
       · generate_custom_dataset     — fully parameterised substrate + VNR stream
       · generate_rl_training_dataset — substrate + v2 stream tuned for RL training
       · generate_stress_dataset     — high-density stress-test dataset

2. VirneDatasetGenerator
   ─ Thin wrapper around virne's Generator.generate_dataset.
   ─ Useful when you need GML persistence and virne's attribute system.
"""

from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from src.generators_v2.substrate_generator import generate_substrate, generate_substrate_virne
from src.generators_v2.vnr_generator import (
    generate_vnr_stream_v2,
    generate_vnr_stream_virne,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DatasetGeneratorV2 — extended version of existing DatasetGenerator
# ══════════════════════════════════════════════════════════════════════════════

class DatasetGeneratorV2:
    """
    Centralised dataset generator for all VNE experiments.

    Backward-compatible with ``src.generators.dataset_generator.DatasetGenerator``:
      - ``generate_fig6_dataset`` signature unchanged (extra kwargs forwarded)
      - ``generate_fig7_dataset`` signature unchanged
      - ``generate_fig8_dataset`` signature unchanged
      - ``generate_all``          still works

    New generators:
      - ``generate_custom_dataset``      — fully user-controlled
      - ``generate_rl_training_dataset`` — tuned for PPO / REINFORCE training
      - ``generate_stress_dataset``      — high-load stress test

    Parameters
    ----------
    base_dir : str
        Root directory for all dataset outputs.
    """

    def __init__(self, base_dir: str = "dataset"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    # ─── internal ─────────────────────────────────────────────────────────────

    def _save_metadata(self, experiment_name: str, metadata: dict) -> None:
        meta_path = os.path.join(self.base_dir, experiment_name, "metadata.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadata saved: {meta_path}")

    def _header(self, title: str, replicas: Optional[int] = None) -> None:
        suffix = f" ({replicas} replicas)" if replicas else ""
        print(f"\n{'='*60}")
        print(f"GENERATING {title.upper()}{suffix}")
        print(f"{'='*60}")

    def _footer(self, title: str, exp_dir: str) -> None:
        print(f"\n{'='*60}")
        print(f"[OK] {title.upper()} COMPLETE")
        print(f"[OK] Location: {exp_dir}")
        print(f"{'='*60}\n")

    # ─── backward-compat experiments ──────────────────────────────────────────

    def generate_fig6_dataset(
        self,
        vnode_range: List[int] = [2, 4, 6, 8],
        num_domains: int = 4,
        num_vnrs_range: Tuple[int, int] = (200, 200),
        substrate_nodes_range: Tuple[int, int] = (100, 100),
        p_intra: float = 0.5,
        p_inter: float = 0.1,
        substrate_cpu_range: Tuple[int, int] = (100, 300),
        substrate_bw_range: Tuple[int, int] = (1000, 3000),
        node_cost_range: Tuple[float, float] = (1, 10),
        inter_domain_bw_cost: Tuple[float, float] = (5, 15),
        vnr_cpu_range: Tuple[int, int] = (50, 200),
        vnr_bw_range: Tuple[int, int] = (500, 2000),
        max_lifetime: int = 1000,
        avg_inter_arrival: float = 1.0,
        num_replicas: int = 10,
        base_seed: int = 42,
        # NEW: optional extra attributes ─────────────────────────────────────
        substrate_memory_range: Optional[Tuple[int, int]] = None,
        vnr_memory_range: Optional[Tuple[int, int]] = None,
        latency_range: Optional[Tuple[float, float]] = None,
    ) -> dict:
        """
        Generate dataset for Fig6 (impact of virtual node count).

        Each replica now produces **one** mixed-size VNR stream whose node counts
        are drawn uniformly from ``[min(vnode_range), max(vnode_range)]`` rather
        than one separate file per fixed count.  This reflects the real training
        scenario where a single dataset contains VNRs of varying sizes.

        ``metadata["vnr_min_nodes"]`` and ``metadata["vnr_max_nodes"]`` record
        the range used; ``metadata["vnode_range"]`` is kept for backwards-compat
        reference only.
        """
        exp_dir = os.path.join(self.base_dir, "fig6")
        os.makedirs(exp_dir, exist_ok=True)
        self._header("FIG6 DATASET", num_replicas)

        min_vnodes = min(vnode_range)
        max_vnodes = max(vnode_range)

        replicas = []
        rng = random.Random(base_seed)

        for replica_id in range(num_replicas):
            replica_seed = base_seed + replica_id * 1000
            local_rng = random.Random(replica_seed)

            num_nodes_total = local_rng.randint(*substrate_nodes_range)
            num_vnrs = local_rng.randint(*num_vnrs_range)

            replica_dir = os.path.join(exp_dir, f"replica_{replica_id}")
            os.makedirs(replica_dir, exist_ok=True)

            print(f"\n[Replica {replica_id + 1}/{num_replicas}] "
                  f"nodes={num_nodes_total}, vnrs={num_vnrs}, "
                  f"vnode_range=[{min_vnodes},{max_vnodes}], seed={replica_seed}")

            substrate_path = os.path.join(replica_dir, "substrate.json")
            generate_substrate(
                num_domains=num_domains,
                num_nodes_total=num_nodes_total,
                p_intra=p_intra, p_inter=p_inter,
                cpu_range=substrate_cpu_range,
                bw_range=substrate_bw_range,
                node_cost_range=node_cost_range,
                inter_domain_bw_cost=inter_domain_bw_cost,
                memory_range=substrate_memory_range,
                latency_range=latency_range,
                seed=replica_seed,
                export_path=substrate_path,
            )

            # Single mixed-range VNR stream per replica
            vnr_path = os.path.join(replica_dir, "vnr_stream.json")
            generate_vnr_stream_v2(
                num_vnrs=num_vnrs,
                num_domains=num_domains,
                min_vnodes=min_vnodes,
                max_vnodes=max_vnodes,
                cpu_range=vnr_cpu_range,
                bw_range=vnr_bw_range,
                max_lifetime=max_lifetime,
                avg_inter_arrival=avg_inter_arrival,
                memory_range=vnr_memory_range,
                latency_range=latency_range,
                export_path=vnr_path,
                seed=replica_seed + 1,
            )

            replicas.append({
                "replica_id": replica_id,
                "seed": replica_seed,
                "substrate_nodes": num_nodes_total,
                "num_vnrs": num_vnrs,
                "substrate_path": substrate_path,
                "vnr_path": vnr_path,
            })

        metadata = {
            "experiment": "fig6",
            "description": "Impact of Virtual Nodes (mixed-range VNR stream)",
            "vnode_range": vnode_range,         # kept for reference
            "vnr_min_nodes": min_vnodes,
            "vnr_max_nodes": max_vnodes,
            "num_domains": num_domains,
            "num_replicas": num_replicas,
            "base_seed": base_seed,
            "replicas": replicas,
        }
        self._save_metadata("fig6", metadata)
        self._footer("FIG6 DATASET", exp_dir)
        return metadata

    def generate_fig7_dataset(
        self,
        domain_range: List[int] = [4, 6, 8],
        num_vnodes: int = 4,
        num_vnrs: int = 300,
        substrate_seed_base: int = 200,
        vnr_seed_base: int = 300,
    ) -> dict:
        """Generate dataset for Fig7 (embedding overhead vs. domain count)."""
        exp_dir = os.path.join(self.base_dir, "fig7")
        os.makedirs(exp_dir, exist_ok=True)
        self._header("FIG7 DATASET")

        substrate_configs: Dict[str, str] = {}
        vnr_configs: Dict[str, str] = {}

        for idx, num_domains in enumerate(domain_range):
            print(f"\n[{idx + 1}/{len(domain_range)}] {num_domains} domains")

            substrate_path = os.path.join(exp_dir, f"substrate_{num_domains}domains.json")
            generate_substrate(
                num_domains=num_domains, num_nodes_total=100,
                p_intra=0.5, p_inter=0.1,
                cpu_range=(100, 300), bw_range=(1000, 3000),
                node_cost_range=(1, 10), inter_domain_bw_cost=(5, 15),
                seed=substrate_seed_base + num_domains,
                export_path=substrate_path,
            )
            substrate_configs[f"{num_domains}domains"] = substrate_path

            vnr_path = os.path.join(exp_dir, f"vnr_{num_domains}domains.json")
            generate_vnr_stream_v2(
                num_vnrs=num_vnrs, num_domains=num_domains,
                max_lifetime=300, avg_inter_arrival=1.0,
                export_path=vnr_path,
                seed=vnr_seed_base + num_domains,
            )
            vnr_configs[f"{num_domains}domains"] = vnr_path

        metadata = {
            "experiment": "fig7",
            "description": "Embedding Overhead vs Number of Domains",
            "domain_range": domain_range,
            "num_vnodes": num_vnodes,
            "num_vnrs": num_vnrs,
            "substrate_configs": substrate_configs,
            "vnr_configs": vnr_configs,
        }
        self._save_metadata("fig7", metadata)
        self._footer("FIG7 DATASET", exp_dir)
        return metadata

    def generate_fig8_dataset(
        self,
        num_vnodes: int = 4,
        num_domains: int = 4,
        num_vnrs: int = 300,
        substrate_seed: int = 400,
        vnr_seed: int = 500,
    ) -> dict:
        """Generate dataset for Fig8 (VNR acceptance rate over time)."""
        exp_dir = os.path.join(self.base_dir, "fig8")
        os.makedirs(exp_dir, exist_ok=True)
        self._header("FIG8 DATASET")

        substrate_path = os.path.join(exp_dir, "substrate.json")
        generate_substrate(
            num_domains=num_domains, num_nodes_total=100,
            p_intra=0.5, p_inter=0.1,
            cpu_range=(100, 300), bw_range=(1000, 3000),
            node_cost_range=(1, 10), inter_domain_bw_cost=(5, 15),
            seed=substrate_seed, export_path=substrate_path,
        )

        vnr_path = os.path.join(exp_dir, "vnr_stream.json")
        generate_vnr_stream_v2(
            num_vnrs=num_vnrs, num_domains=num_domains,
            max_lifetime=300, avg_inter_arrival=1.0,
            export_path=vnr_path, seed=vnr_seed,
        )

        metadata = {
            "experiment": "fig8",
            "description": "VNR Acceptance Rate Over Time",
            "num_vnodes": num_vnodes,
            "num_domains": num_domains,
            "num_vnrs": num_vnrs,
            "substrate_path": substrate_path,
            "vnr_path": vnr_path,
        }
        self._save_metadata("fig8", metadata)
        self._footer("FIG8 DATASET", exp_dir)
        return metadata

    def generate_all(self) -> dict:
        """Generate all three standard experiment datasets."""
        print(f"\n{'='*70}")
        print(" GENERATING ALL EXPERIMENT DATASETS ".center(70, "="))
        print(f"{'='*70}")
        results = {
            "fig6": self.generate_fig6_dataset(),
            "fig7": self.generate_fig7_dataset(),
            "fig8": self.generate_fig8_dataset(),
        }
        print(f"\n{'='*70}")
        print(" ALL DATASETS GENERATED SUCCESSFULLY ".center(70, "="))
        print(f"{'='*70}")
        print(f"\n  Datasets location: {self.base_dir}/\n")
        return results

    # ─── NEW experiments ───────────────────────────────────────────────────────

    def generate_custom_dataset(
        self,
        name: str = "custom",
        # Substrate ─────────────────────────────────────────────────────────
        num_domains: int = 4,
        num_nodes_total: int = 80,
        p_intra: float = 0.5,
        p_inter: float = 0.05,
        substrate_cpu_range: Tuple[int, int] = (100, 300),
        substrate_bw_range: Tuple[int, int] = (1000, 3000),
        node_cost_range: Tuple[float, float] = (1, 10),
        inter_domain_bw_cost: Tuple[float, float] = (5, 15),
        substrate_memory_range: Optional[Tuple[int, int]] = None,
        substrate_gpu_range: Optional[Tuple[int, int]] = None,
        substrate_latency_range: Optional[Tuple[float, float]] = None,
        # VNR stream ─────────────────────────────────────────────────────────
        num_vnrs: int = 500,
        min_vnodes: int = 2,
        max_vnodes: int = 10,
        vnr_cpu_range: Tuple[int, int] = (5, 30),
        vnr_bw_range: Tuple[int, int] = (10, 50),
        max_lifetime: int = 300,
        avg_inter_arrival: float = 1.0,
        vnr_memory_range: Optional[Tuple[int, int]] = None,
        vnr_gpu_range: Optional[Tuple[int, int]] = None,
        vnr_latency_range: Optional[Tuple[float, float]] = None,
        max_latency_range: Optional[Tuple[float, float]] = None,
        hot_domain_prob: float = 0.6,
        # ────────────────────────────────────────────────────────────────────
        seed: int = 42,
    ) -> dict:
        """
        Fully parameterised dataset generator.

        Creates one substrate and one VNR stream with complete control over all
        parameters including optional memory, GPU, and latency attributes.

        Parameters
        ----------
        name : str
            Sub-directory name under base_dir.  Allows multiple custom datasets.
        (all other params)
            See module-level generate_substrate / generate_vnr_stream_v2 docs.

        Returns
        -------
        dict  Metadata dict with paths and all generation parameters.
        """
        exp_dir = os.path.join(self.base_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        self._header(f"CUSTOM DATASET [{name}]")

        substrate_path = os.path.join(exp_dir, "substrate.json")
        print("\n[1/2] Substrate …")
        generate_substrate(
            num_domains=num_domains,
            num_nodes_total=num_nodes_total,
            p_intra=p_intra,
            p_inter=p_inter,
            cpu_range=substrate_cpu_range,
            bw_range=substrate_bw_range,
            node_cost_range=node_cost_range,
            inter_domain_bw_cost=inter_domain_bw_cost,
            memory_range=substrate_memory_range,
            gpu_range=substrate_gpu_range,
            latency_range=substrate_latency_range,
            seed=seed,
            export_path=substrate_path,
        )

        vnr_path = os.path.join(exp_dir, "vnr_stream.json")
        print("[2/2] VNR stream …")
        generate_vnr_stream_v2(
            num_vnrs=num_vnrs,
            num_domains=num_domains,
            min_vnodes=min_vnodes,
            max_vnodes=max_vnodes,
            cpu_range=vnr_cpu_range,
            bw_range=vnr_bw_range,
            max_lifetime=max_lifetime,
            avg_inter_arrival=avg_inter_arrival,
            memory_range=vnr_memory_range,
            gpu_range=vnr_gpu_range,
            latency_range=vnr_latency_range,
            max_latency_range=max_latency_range,
            hot_domain_prob=hot_domain_prob,
            seed=seed + 1,
            export_path=vnr_path,
        )

        metadata = {
            "experiment": name,
            "substrate": {
                "path": substrate_path,
                "num_domains": num_domains,
                "num_nodes_total": num_nodes_total,
                "cpu_range": list(substrate_cpu_range),
                "bw_range": list(substrate_bw_range),
            },
            "vnr_stream": {
                "path": vnr_path,
                "num_vnrs": num_vnrs,
                "min_vnodes": min_vnodes,
                "max_vnodes": max_vnodes,
                "cpu_range": list(vnr_cpu_range),
                "bw_range": list(vnr_bw_range),
            },
            "seed": seed,
        }
        self._save_metadata(name, metadata)
        self._footer(f"CUSTOM DATASET [{name}]", exp_dir)
        return metadata

    def generate_rl_training_dataset(
        self,
        num_domains: int = 4,
        num_nodes_total: int = 80,
        num_vnrs: int = 1000,
        min_vnodes: int = 2,
        max_vnodes: int = 10,
        substrate_cpu_range: Tuple[int, int] = (100, 300),
        substrate_bw_range: Tuple[int, int] = (1000, 3000),
        vnr_cpu_range: Tuple[int, int] = (10, 80),
        vnr_bw_range: Tuple[int, int] = (50, 500),
        max_lifetime: int = 500,
        avg_inter_arrival: float = 1.0,
        seed: int = 999,
    ) -> dict:
        """
        Pre-tuned dataset for RL agent training (PPO / REINFORCE).

        VNR node counts vary uniformly from ``min_vnodes`` to ``max_vnodes``
        within a single stream, exposing the agent to requests of all sizes during
        training and improving generalisation.

        Returns
        -------
        dict with keys ``substrate_path`` and ``vnr_path``.
        """
        return self.generate_custom_dataset(
            name="rl_training",
            num_domains=num_domains,
            num_nodes_total=num_nodes_total,
            p_intra=0.6,
            p_inter=0.05,
            substrate_cpu_range=substrate_cpu_range,
            substrate_bw_range=substrate_bw_range,
            num_vnrs=num_vnrs,
            min_vnodes=min_vnodes,
            max_vnodes=max_vnodes,
            vnr_cpu_range=vnr_cpu_range,
            vnr_bw_range=vnr_bw_range,
            max_lifetime=max_lifetime,
            avg_inter_arrival=avg_inter_arrival,
            hot_domain_prob=0.7,
            seed=seed,
        )

    def generate_stress_dataset(
        self,
        num_domains: int = 6,
        num_nodes_total: int = 150,
        num_vnrs: int = 2000,
        seed: int = 77,
    ) -> dict:
        """
        High-density stress-test dataset.

        Creates a large substrate with many domains and a long VNR stream of
        heavy, bursty requests.  Useful for scalability benchmarking.
        """
        return self.generate_custom_dataset(
            name="stress_test",
            num_domains=num_domains,
            num_nodes_total=num_nodes_total,
            p_intra=0.6,
            p_inter=0.08,
            substrate_cpu_range=(200, 600),
            substrate_bw_range=(2000, 8000),
            num_vnrs=num_vnrs,
            min_vnodes=4,
            max_vnodes=20,
            vnr_cpu_range=(30, 200),
            vnr_bw_range=(100, 1000),
            max_lifetime=500,
            avg_inter_arrival=0.5,
            hot_domain_prob=0.5,
            seed=seed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. VirneDatasetGenerator — wrapper around virne's Generator
# ══════════════════════════════════════════════════════════════════════════════

class VirneDatasetGenerator:
    """
    Config-driven dataset generator that delegates to virne's Generator class.

    Use this when you need:
      - virne's attribute configuration system (distribution types, extrema, …)
      - GML file persistence with full attribute metadata
      - Waxman / Barabási–Albert topologies for the physical network

    Parameters
    ----------
    base_dir : str
        Root output directory.
    """

    def __init__(self, base_dir: str = "dataset_virne"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def generate(
        self,
        config: Dict[str, Any],
        name: str = "virne_dataset",
        p_net: bool = True,
        v_nets: bool = True,
        save: bool = True,
        changeable: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Generate a virne-native dataset.

        Parameters
        ----------
        config : dict
            Full virne configuration with ``p_net_setting`` and ``v_sim_setting``.
        name : str
            Sub-directory name under base_dir.
        p_net : bool
            Generate physical network.
        v_nets : bool
            Generate virtual network stream.
        save : bool
            Persist to disk as GML / YAML.
        changeable : bool
            Use the 4-stage changeable VNR generator.
        seed : int or None
            Global random seed.

        Returns
        -------
        Tuple[PhysicalNetwork or None, VirtualNetworkRequestSimulator or None]
        """
        try:
            from virne.network.dataset_generator import Generator
        except ImportError as exc:
            raise ImportError(
                "virne is not importable. Add virne-main to sys.path:\n"
                "  import sys; sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')"
            ) from exc

        dataset_dir = os.path.join(self.base_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)

        physical_net = None
        v_net_sim = None

        if p_net and "p_net_setting" in config:
            physical_net = Generator.generate_p_net_dataset_from_config(config, save=save)
            if save:
                p_dir = os.path.join(dataset_dir, "p_net")
                physical_net.save_dataset(p_dir)
                print(f"[OK] PhysicalNetwork saved: {p_dir}")

        if v_nets and "v_sim_setting" in config:
            if changeable:
                v_net_sim = Generator.generate_changeable_v_nets_dataset_from_config(
                    config, save=save
                )
            else:
                v_net_sim = Generator.generate_v_nets_dataset_from_config(
                    config, save=save
                )
            if save and not (changeable and save):
                v_dir = os.path.join(dataset_dir, "v_nets")
                v_net_sim.save_dataset(v_dir)
                print(f"[OK] VirtualNetworkRequestSimulator saved: {v_dir}")

        return physical_net, v_net_sim

    @staticmethod
    def load(dataset_dir: str, p_net_subdir: str = "p_net", v_nets_subdir: str = "v_nets"):
        """
        Load a previously-saved virne dataset.

        Parameters
        ----------
        dataset_dir : str
            Top-level dataset directory (same as passed to ``generate``).
        p_net_subdir : str
            Sub-directory containing the physical network GML.
        v_nets_subdir : str
            Sub-directory containing the VNR GML files and YAML events.

        Returns
        -------
        Tuple[PhysicalNetwork or None, VirtualNetworkRequestSimulator or None]
        """
        try:
            from virne.network.physical_network import PhysicalNetwork
            from virne.network.virtual_network_request_simulator import (
                VirtualNetworkRequestSimulator,
            )
        except ImportError as exc:
            raise ImportError(
                "virne is not importable. Add virne-main to sys.path."
            ) from exc

        p_net, v_sim = None, None

        p_path = os.path.join(dataset_dir, p_net_subdir)
        if os.path.isdir(p_path):
            p_net = PhysicalNetwork.load_dataset(p_path)
            print(f"[OK] PhysicalNetwork loaded from {p_path}")

        v_path = os.path.join(dataset_dir, v_nets_subdir)
        if os.path.isdir(v_path):
            v_sim = VirtualNetworkRequestSimulator.load_dataset(v_path)
            print(f"[OK] VirtualNetworkRequestSimulator loaded from {v_path}")

        return p_net, v_sim


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="generators_v2 dataset generator")
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["fig6", "fig7", "fig8", "all", "rl", "stress", "custom"],
        default=["all"],
    )
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-vnrs", type=int, default=300)
    args = parser.parse_args()

    gen = DatasetGeneratorV2(base_dir=args.output_dir)

    if "all" in args.experiments:
        gen.generate_all()
    else:
        for exp in args.experiments:
            if exp == "fig6":
                gen.generate_fig6_dataset()
            elif exp == "fig7":
                gen.generate_fig7_dataset()
            elif exp == "fig8":
                gen.generate_fig8_dataset()
            elif exp == "rl":
                gen.generate_rl_training_dataset(seed=args.seed)
            elif exp == "stress":
                gen.generate_stress_dataset(seed=args.seed)
            elif exp == "custom":
                gen.generate_custom_dataset(
                    name="custom", num_vnrs=args.num_vnrs, seed=args.seed
                )


if __name__ == "__main__":
    main()
