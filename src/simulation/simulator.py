# src/simulation/simulator.py
"""
VNR Simulation Engine
Core simulation logic extracted from experiment.py for reusability.
"""

from collections import deque
from datetime import datetime
from src.utils.graph_utils import copy_substrate, release_node, release_path
from src.evaluation.metrics import cost_of_embedding, revenue_of_vnr


class VNRSimulator:
    """
    Core VNR embedding simulator.
    Handles resource management, lifetime tracking, and metrics collection.
    """
    
    def __init__(self, substrate_graph):
        """
        Initialize simulator with substrate network.
        
        Args:
            substrate_graph: NetworkX substrate graph
        """
        self.substrate_original = substrate_graph
        self.substrate = copy_substrate(substrate_graph)
        self.active_vnrs = deque()  # (vnr, mapping, link_paths, departure_time)
        self.current_time = 0
        
        # Metrics
        self.success_count = 0
        self.total_cost = 0
        self.total_revenue = 0
        self.execution_times = []
    
    def reset(self):
        """Reset simulator to initial state."""
        self.substrate = copy_substrate(self.substrate_original)
        self.active_vnrs = deque()
        self.current_time = 0
        self.success_count = 0
        self.total_cost = 0
        self.total_revenue = 0
        self.execution_times = []
    
    def release_vnr_resources(self, vnr_graph, mapping, link_paths):
        """
        Release resources consumed by a VNR.
        
        Args:
            vnr_graph: The VNR graph
            mapping: Node mapping dict {vnode: snode}
            link_paths: Link mapping dict {(u,v): path}
        """
        # Release node resources
        for vnode, snode in mapping.items():
            cpu_req = vnr_graph.nodes[vnode]['cpu']
            release_node(self.substrate, snode, cpu_req)
        
        # Release link resources
        for (u, v), path in link_paths.items():
            bw_req = vnr_graph.edges[u, v]['bw']
            release_path(self.substrate, path, bw_req)
    
    def process_departures(self, current_time, verbose=False):
        """
        Release expired VNRs at current time.
        
        Args:
            current_time: Current simulation time
            verbose: If True, print release messages
            
        Returns:
            Number of VNRs released
        """
        released_count = 0
        
        while self.active_vnrs and self.active_vnrs[0][3] <= current_time:
            expired_vnr, exp_mapping, exp_paths, dep_time = self.active_vnrs.popleft()
            self.release_vnr_resources(expired_vnr, exp_mapping, exp_paths)
            released_count += 1
            
            if verbose:
                print(f"   [t={current_time}] Released VNR (departed at {dep_time})")
        
        return released_count
    
    def embed_vnr(self, vnr_graph, embedding_algorithm, verbose=False):
        """
        Attempt to embed a single VNR using provided algorithm.
        
        Args:
            vnr_graph: VNR to embed
            embedding_algorithm: Function(substrate, vnr) -> (mapping, link_paths) or None
            verbose: If True, print embedding status
            
        Returns:
            True if successful, False otherwise
        """
        arrival_time = vnr_graph.graph.get('arrival_time', self.current_time)
        
        # Release expired VNRs first
        self.process_departures(arrival_time, verbose)
        
        # Update current time
        self.current_time = arrival_time
        
        # Try to embed
        start_time = datetime.now()
        
        # Work on a copy to avoid modifying original
        working_substrate = copy_substrate(self.substrate)
        result = embedding_algorithm(working_substrate, vnr_graph)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.execution_times.append(elapsed)
        
        if result is None:
            if verbose:
                vnr_id = vnr_graph.graph.get('id', '?')
                print(f"   [t={self.current_time}] VNR {vnr_id} rejected")
            return False
        
        # Extract results
        mapping, link_paths = result
        
        # Calculate cost and revenue
        cost = cost_of_embedding(mapping, link_paths, vnr_graph, working_substrate)
        revenue = revenue_of_vnr(vnr_graph)
        
        self.total_cost += cost
        self.total_revenue += revenue
        self.success_count += 1
        
        # Update substrate with reserved resources
        self.substrate = working_substrate
        
        # Calculate departure time and add to active VNRs
        lifetime = vnr_graph.graph.get('lifetime', 50)
        departure_time = self.current_time + lifetime
        self.active_vnrs.append((vnr_graph, mapping, link_paths, departure_time))
        
        if verbose:
            vnr_id = vnr_graph.graph.get('id', '?')
            print(f"   [t={self.current_time}] VNR {vnr_id} embedded "
                  f"(departs at {departure_time}, cost={cost:.2f})")
        
        return True
    
    def simulate_stream(self, vnr_stream, embedding_algorithm, verbose=False):
        """
        Simulate VNR stream with lifetime management.
        
        Args:
            vnr_stream: List of VNR graphs
            embedding_algorithm: Function(substrate, vnr) -> (mapping, link_paths) or None
            verbose: If True, print detailed progress
            
        Returns:
            Dictionary containing simulation metrics
        """
        self.reset()
        
        for vnr in vnr_stream:
            self.embed_vnr(vnr, embedding_algorithm, verbose)
        
        # Release all remaining active VNRs
        while self.active_vnrs:
            expired_vnr, exp_mapping, exp_paths, dep_time = self.active_vnrs.popleft()
            self.release_vnr_resources(expired_vnr, exp_mapping, exp_paths)
        
        return self.get_metrics(len(vnr_stream))
    
    def get_metrics(self, total_vnrs):
        """
        Calculate summary metrics.
        
        Args:
            total_vnrs: Total number of VNRs processed
            
        Returns:
            Dictionary of metrics
        """
        acceptance_ratio = self.success_count / total_vnrs if total_vnrs else 0
        avg_cost = self.total_cost / self.success_count if self.success_count > 0 else 0
        avg_revenue = self.total_revenue / self.success_count if self.success_count > 0 else 0
        avg_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        cost_revenue_ratio = self.total_cost / self.total_revenue if self.total_revenue > 0 else 0
        
        return {
            'acceptance_ratio': acceptance_ratio,
            'avg_cost': avg_cost,
            'avg_revenue': avg_revenue,
            'cost_revenue_ratio': cost_revenue_ratio,
            'avg_execution_time': avg_time,
            'successful_embeddings': self.success_count,
            'total_vnrs': total_vnrs,
            'total_cost': self.total_cost,
            'total_revenue': self.total_revenue
        }


class BatchedVNRSimulator(VNRSimulator):
    """
    VNR Simulator with time-window batching support.
    Extends base simulator for batch processing algorithms.
    """
    
    def __init__(self, substrate_graph, window_size=10, max_queue_delay=30):
        """
        Initialize batched simulator.
        
        Args:
            substrate_graph: NetworkX substrate graph
            window_size: Duration of time window
            max_queue_delay: Maximum queue delay before expiration
        """
        super().__init__(substrate_graph)
        self.window_size = window_size
        self.max_queue_delay = max_queue_delay
        self.expired_in_queue = 0
    
    def reset(self):
        """Reset simulator including batch-specific metrics."""
        super().reset()
        self.expired_in_queue = 0
    
    def simulate_batched_stream(self, vnr_stream, batch_algorithm, verbose=False):
        """
        Simulate VNR stream with time-window batching.
        
        Args:
            vnr_stream: List of VNR graphs
            batch_algorithm: Function(substrate, batch) -> (accepted, rejected)
                           where batch = [(vnr, revenue), ...]
            verbose: If True, print detailed progress
            
        Returns:
            Dictionary containing simulation metrics
        """
        from src.simulation.batching import TimeWindowBatcher
        
        self.reset()
        batcher = TimeWindowBatcher(
            window_size=self.window_size,
            max_queue_delay=self.max_queue_delay
        )
        
        time_series_data = []
        processed_vnrs = 0
        window_idx = 0
        window_expired_sum = 0
        
        for vnr in vnr_stream:
            arrival_time = vnr.graph.get('arrival_time', self.current_time)
            
            # Release expired VNRs from substrate
            self.process_departures(arrival_time, verbose)
            
            self.current_time = arrival_time
            
            # Remove expired VNRs from queue
            expired_count = batcher.remove_expired_requests(self.current_time)
            if expired_count > 0:
                self.expired_in_queue += expired_count
                processed_vnrs += expired_count
                window_expired_sum += expired_count
                if verbose:
                    print(f"   [t={self.current_time}] Removed {expired_count} "
                          f"expired VNR(s) from queue")
            
            # Add VNR to time window
            should_process = batcher.add_request(vnr, self.current_time)
            
            # Process batch if window is complete
            if should_process:
                accepted_count, rejected_count = self._process_batch(batcher, batch_algorithm, verbose)
                processed_vnrs += (accepted_count + rejected_count)
                
                window_metrics = self.get_metrics(processed_vnrs)
                window_metrics['time'] = self.current_time
                window_metrics['window_idx'] = window_idx
                window_metrics['expired_in_queue'] = self.expired_in_queue
                window_metrics['window_accepted'] = accepted_count
                window_metrics['window_rejected'] = rejected_count
                window_metrics['window_expired'] = window_expired_sum
                
                time_series_data.append(window_metrics)
                
                window_idx += 1
                window_expired_sum = 0
        
        # Process any remaining VNRs in queue
        if batcher.has_pending_requests():
            expired_count = batcher.remove_expired_requests(self.current_time)
            if expired_count > 0:
                self.expired_in_queue += expired_count
                processed_vnrs += expired_count
                window_expired_sum += expired_count
                if verbose:
                    print(f"   [t={self.current_time}] Removed {expired_count} "
                          f"expired VNR(s) from final queue")
            
            batch = batcher.get_pending_batch(self.current_time)
            accepted_count, rejected_count = 0, 0
            if batch:
                if verbose:
                    print(f"   [t={self.current_time}] Processing final batch of {len(batch)} VNRs")
                accepted_count, rejected_count = self._embed_batch(batch, batch_algorithm, verbose)
                processed_vnrs += (accepted_count + rejected_count)
                
            window_metrics = self.get_metrics(processed_vnrs)
            window_metrics['time'] = self.current_time
            window_metrics['window_idx'] = window_idx
            window_metrics['expired_in_queue'] = self.expired_in_queue
            window_metrics['window_accepted'] = accepted_count
            window_metrics['window_rejected'] = rejected_count
            window_metrics['window_expired'] = window_expired_sum
            
            time_series_data.append(window_metrics)
        
        # Release all remaining active VNRs
        while self.active_vnrs:
            expired_vnr, exp_mapping, exp_paths, dep_time = self.active_vnrs.popleft()
            self.release_vnr_resources(expired_vnr, exp_mapping, exp_paths)
        
        # Get metrics with expired count
        metrics = self.get_metrics(len(vnr_stream))
        metrics['expired_in_queue'] = self.expired_in_queue
        metrics['rejected_by_algorithm'] = (len(vnr_stream) - 
                                           self.success_count - 
                                           self.expired_in_queue)
        metrics['time_series'] = time_series_data
        
        return metrics
    
    def _process_batch(self, batcher, batch_algorithm, verbose):
        """Process a complete time window batch."""
        batch = batcher.get_batch_and_reset(self.current_time)
        
        if not batch:
            if verbose:
                print(f"   [t={self.current_time}] Batch empty after filtering")
            return 0, 0
        
        if verbose:
            print(f"   [t={self.current_time}] Processing batch of {len(batch)} VNRs")
        
        return self._embed_batch(batch, batch_algorithm, verbose)
    
    def _embed_batch(self, batch, batch_algorithm, verbose):
        """Embed a batch of VNRs using batch algorithm."""
        start_time = datetime.now()
        accepted, rejected = batch_algorithm(self.substrate, batch)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Process accepted VNRs
        for vnr_graph, mapping, link_paths in accepted:
            self.success_count += 1
            self.execution_times.append(elapsed / len(batch))
            
            # Calculate cost and revenue
            cost = cost_of_embedding(mapping, link_paths, vnr_graph, self.substrate)
            revenue = revenue_of_vnr(vnr_graph)
            
            self.total_cost += cost
            self.total_revenue += revenue
            
            # Add to active VNRs
            lifetime = vnr_graph.graph.get('lifetime', 50)
            departure_time = self.current_time + lifetime
            self.active_vnrs.append((vnr_graph, mapping, link_paths, departure_time))
            
            if verbose:
                vnr_id = vnr_graph.graph.get('id', '?')
                print(f"      VNR {vnr_id} embedded (departs at {departure_time}, cost={cost:.2f})")
        
        # Log rejected VNRs
        if verbose and rejected:
            print(f"      {len(rejected)} VNRs rejected")
            
        return len(accepted), len(rejected)