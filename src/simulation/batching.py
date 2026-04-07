"""
Time Window Batching for VNR Requests
Logic extracted from utils.py - manages time-window batch processing.
"""

from evaluation.metrics import revenue_of_vnr


class TimeWindowBatcher:
    """
    Manages time window batching for VNR requests.
    Collects requests within a time window and processes them together.
    Handles request expiration based on maximum queue delay.
    """
    
    def __init__(self, window_size=10, max_queue_delay=50):
        """
        Initialize time window batcher.
        
        Args:
            window_size: Duration of time window to collect requests
            max_queue_delay: Maximum time a request can wait in queue
        """
        self.window_size = window_size
        self.max_queue_delay = max_queue_delay
        self.request_queue = []  # Stores (vnr, arrival_time)
        self.current_window_end = None
    
    def add_request(self, vnr, current_time):
        """
        Add a VNR request to the queue.
        
        Args:
            vnr: VNR graph to embed
            current_time: Current simulation time
            
        Returns:
            bool: True if window should be processed, False otherwise
        """
        # Initialize window on first request
        if self.current_window_end is None:
            self.current_window_end = current_time + self.window_size
        
        # Add request to queue
        self.request_queue.append((vnr, current_time))
        
        # Check if window is complete
        return current_time >= self.current_window_end
    
    def remove_expired_requests(self, current_time):
        """
        Remove VNR requests that have exceeded max queue delay.
        Should be called periodically as simulation time advances.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            int: Number of expired requests removed
        """
        initial_count = len(self.request_queue)
        
        # Filter out expired requests
        self.request_queue = [
            (vnr, arrival) for vnr, arrival in self.request_queue
            if (current_time - arrival) <= self.max_queue_delay
        ]
        
        expired_count = initial_count - len(self.request_queue)
        return expired_count
    
    def get_batch_and_reset(self, current_time):
        """
        Get current batch and reset window.
        Drops requests that exceed max queue delay.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of (vnr, revenue) tuples ready for embedding
        """
        # Filter out expired requests
        valid_requests = [
            (vnr, arrival) for vnr, arrival in self.request_queue
            if (current_time - arrival) <= self.max_queue_delay
        ]
        
        # Calculate revenue for each VNR
        batch = []
        for vnr, arrival_time in valid_requests:
            revenue = revenue_of_vnr(vnr)
            batch.append((vnr, revenue))
        
        # Reset for next window
        self.request_queue = []
        self.current_window_end = current_time + self.window_size
        
        return batch
    
    def has_pending_requests(self):
        """Check if there are requests waiting in queue."""
        return len(self.request_queue) > 0
    
    def get_pending_batch(self, current_time):
        """
        Get all pending requests at end of simulation.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of (vnr, revenue) tuples
        """
        return self.get_batch_and_reset(current_time)