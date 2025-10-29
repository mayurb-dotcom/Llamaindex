"""Resource monitoring for large document processing"""
import psutil
import time
from rich.console import Console
from rich.table import Table
from threading import Thread, Event

console = Console()

class ResourceMonitor:
    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.monitoring = Event()
        self.peak_memory = 0
        self.peak_cpu = 0
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring.set()
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring.clear()
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Monitor resource usage"""
        while self.monitoring.is_set():
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            cpu = psutil.cpu_percent(interval=1)
            
            self.peak_memory = max(self.peak_memory, memory)
            self.peak_cpu = max(self.peak_cpu, cpu)
            
            time.sleep(self.update_interval)
            
    def get_report(self):
        """Get resource usage report"""
        table = Table(title="Resource Usage Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Peak Memory Usage", f"{self.peak_memory:.2f} MB")
        table.add_row("Peak CPU Usage", f"{self.peak_cpu:.1f}%")
        
        return table