"""
Real-time Progress and Performance Monitor for MatSci-ML Studio
"""

import time
import threading
from typing import Dict, List, Optional, Any

# Optional psutil import with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed, system monitoring will be disabled")
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QLabel, QProgressBar, QTextEdit, QPushButton,
                            QTableWidget, QTableWidgetItem, QSplitter,
                            QTabWidget, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QColor
import numpy as np

# matplotlib import with exception handling
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed, performance charts will be disabled")

class SystemMonitorWorker(QThread):
    """System performance monitoring worker thread"""
    
    # Signals
    system_stats_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.update_interval = 1.0  # Update every 1 second
        
    def run(self):
        """Run monitoring loop"""
        self.running = True
        while self.running:
            try:
                # Collect system statistics
                stats = self.collect_system_stats()
                self.system_stats_updated.emit(stats)
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"System monitor error: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.quit()
        self.wait()
    
    def collect_system_stats(self):
        """Collect system statistics"""
        if not PSUTIL_AVAILABLE:
            # Return dummy stats if psutil is not available
            return {
                'timestamp': time.time(),
                'cpu': {'percent': 0, 'count': 1, 'frequency': 0},
                'memory': {'total': 1, 'available': 1, 'percent': 0, 'used': 0},
                'disk': {'total': 1, 'used': 0, 'free': 1, 'percent': 0},
                'network': {'bytes_sent': 0, 'bytes_recv': 0},
                'process': {'memory_mb': 0, 'cpu_percent': 0}
            }
        
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            # Network information
            network = psutil.net_io_counters()
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            stats = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else 0
                },
                'memory': {
                    'total': memory.total / (1024**3),  # GB
                    'available': memory.available / (1024**3),  # GB
                    'percent': memory.percent,
                    'used': memory.used / (1024**3)  # GB
                },
                'disk': {
                    'total': disk.total / (1024**3),  # GB
                    'used': disk.used / (1024**3),  # GB
                    'free': disk.free / (1024**3),  # GB
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent / (1024**2),  # MB
                    'bytes_recv': network.bytes_recv / (1024**2)  # MB
                },
                'process': {
                    'memory_mb': process_memory.rss / (1024**2),  # MB
                    'cpu_percent': process_cpu
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"Error collecting system stats: {e}")
            return {}

class TaskProgressTracker:
    """Task progress tracker"""
    
    def __init__(self):
        self.tasks = {}  # task_id -> task_info
        self.current_task = None
        
    def start_task(self, task_id: str, task_name: str, total_steps: int = 100):
        """Start new task"""
        self.tasks[task_id] = {
            'name': task_name,
            'start_time': time.time(),
            'total_steps': total_steps,
            'current_step': 0,
            'status': 'running',
            'substeps': [],
            'estimated_time': None,
            'elapsed_time': 0
        }
        self.current_task = task_id
        
    def update_progress(self, task_id: str, current_step: int, status_message: str = ""):
        """Update task progress"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task['current_step'] = current_step
            task['elapsed_time'] = time.time() - task['start_time']
            
            # Estimate remaining time
            if current_step > 0:
                time_per_step = task['elapsed_time'] / current_step
                remaining_steps = task['total_steps'] - current_step
                task['estimated_time'] = remaining_steps * time_per_step
            
            # Add substep record
            if status_message:
                task['substeps'].append({
                    'step': current_step,
                    'message': status_message,
                    'timestamp': time.time()
                })
                
    def complete_task(self, task_id: str, success: bool = True):
        """Complete task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task['status'] = 'completed' if success else 'failed'
            task['end_time'] = time.time()
            task['total_time'] = task['end_time'] - task['start_time']
            
    def get_task_info(self, task_id: str) -> dict:
        """Get task information"""
        return self.tasks.get(task_id, {})
    
    def get_all_tasks(self) -> dict:
        """Get all tasks"""
        return self.tasks

class PerformanceChart(QWidget):
    """Performance chart component"""
    
    def __init__(self, chart_type: str = 'cpu'):
        super().__init__()
        self.chart_type = chart_type
        self.data_history = []
        self.max_points = 60  # Keep 60 data points
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib chart
            self.figure = Figure(figsize=(8, 4))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            
            self.ax = self.figure.add_subplot(111)
            self.line, = self.ax.plot([], [], 'b-', linewidth=2)
            
            # Set chart title and labels
            if self.chart_type == 'cpu':
                self.ax.set_title('CPU Usage', fontsize=12)
                self.ax.set_ylabel('Usage (%)')
                self.ax.set_ylim(0, 100)
            elif self.chart_type == 'memory':
                self.ax.set_title('Memory Usage', fontsize=12)
                self.ax.set_ylabel('Usage (%)')
                self.ax.set_ylim(0, 100)
            
            self.ax.set_xlabel('Time (seconds)')
            self.ax.grid(True, alpha=0.3)
        else:
            # If matplotlib not available, show text prompt
            from PyQt5.QtWidgets import QLabel
            label = QLabel(f"Performance charts require matplotlib\nDisplaying {self.chart_type} data")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        
    def update_data(self, value: float):
        """Update data"""
        self.data_history.append(value)
        
        # Limit data points
        if len(self.data_history) > self.max_points:
            self.data_history.pop(0)
        
        if MATPLOTLIB_AVAILABLE:
            # Update chart
            x_data = list(range(len(self.data_history)))
            self.line.set_data(x_data, self.data_history)
            
            # Adjust x-axis range
            if len(x_data) > 0:
                self.ax.set_xlim(0, max(len(x_data), self.max_points))
            
            # Redraw
            self.canvas.draw()

class PerformanceMonitor(QWidget):
    """Performance monitor main interface"""
    
    # Signals
    performance_alert = pyqtSignal(str, str)  # alert_type, message
    
    def __init__(self):
        super().__init__()
        self.system_monitor = None
        self.task_tracker = TaskProgressTracker()
        self.system_stats_history = []
        
        # Alert status tracking to prevent repeated warnings
        self.cpu_alert_sent = False
        self.memory_alert_sent = False
        # Persistent alert flags - once shown, never show again
        self.cpu_alert_shown_once = False
        self.memory_alert_shown_once = False
        
        self.init_ui()
        self.start_monitoring()
        
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 Real-time Performance Monitor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2196F3; margin: 10px; }")
        layout.addWidget(title)
        
        # Create tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. System monitor
        self.create_system_monitor_tab()
        
        # 2. Task progress
        self.create_task_progress_tab()
        
        # 3. Performance charts
        self.create_performance_charts_tab()
        
        # Control buttons
        self.create_control_buttons(layout)
        
    def create_system_monitor_tab(self):
        """Create system monitor tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Real-time statistics
        stats_group = QGroupBox("🖥️ System Resources")
        stats_layout = QVBoxLayout(stats_group)
        
        # CPU information
        cpu_frame = QFrame()
        cpu_layout = QHBoxLayout(cpu_frame)
        
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_label.setFont(QFont("Arial", 12))
        cpu_layout.addWidget(self.cpu_label)
        
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)
        cpu_layout.addWidget(self.cpu_progress)
        
        stats_layout.addWidget(cpu_frame)
        
        # Memory information
        memory_frame = QFrame()
        memory_layout = QHBoxLayout(memory_frame)
        
        self.memory_label = QLabel("Memory: 0%")
        self.memory_label.setFont(QFont("Arial", 12))
        memory_layout.addWidget(self.memory_label)
        
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        memory_layout.addWidget(self.memory_progress)
        
        stats_layout.addWidget(memory_frame)
        
        # Detailed information
        self.system_details_text = QTextEdit()
        self.system_details_text.setMaximumHeight(200)
        self.system_details_text.setReadOnly(True)
        stats_layout.addWidget(self.system_details_text)
        
        layout.addWidget(stats_group)
        
        # Process monitoring
        process_group = QGroupBox("🔧 Current Process")
        process_layout = QVBoxLayout(process_group)
        
        self.process_info_text = QTextEdit()
        self.process_info_text.setMaximumHeight(150)
        self.process_info_text.setReadOnly(True)
        process_layout.addWidget(self.process_info_text)
        
        layout.addWidget(process_group)
        
        self.tabs.addTab(tab, "System Monitor")
        
    def create_task_progress_tab(self):
        """Create task progress tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Current task
        current_task_group = QGroupBox("🎯 Current Task")
        current_task_layout = QVBoxLayout(current_task_group)
        
        self.current_task_label = QLabel("No running tasks")
        self.current_task_label.setFont(QFont("Arial", 12, QFont.Bold))
        current_task_layout.addWidget(self.current_task_label)
        
        self.current_task_progress = QProgressBar()
        current_task_layout.addWidget(self.current_task_progress)
        
        self.task_status_label = QLabel("")
        current_task_layout.addWidget(self.task_status_label)
        
        self.time_info_label = QLabel("")
        current_task_layout.addWidget(self.time_info_label)
        
        layout.addWidget(current_task_group)
        
        # Task history
        history_group = QGroupBox("📋 Task History")
        history_layout = QVBoxLayout(history_group)
        
        self.task_history_table = QTableWidget()
        self.task_history_table.setColumnCount(5)
        self.task_history_table.setHorizontalHeaderLabels(['Task Name', 'Status', 'Progress', 'Duration', 'Completion Time'])
        history_layout.addWidget(self.task_history_table)
        
        layout.addWidget(history_group)
        
        self.tabs.addTab(tab, "Task Progress")
        
    def create_performance_charts_tab(self):
        """Create performance charts tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create charts container
        charts_splitter = QSplitter(Qt.Vertical)
        
        # CPU chart
        self.cpu_chart = PerformanceChart('cpu')
        charts_splitter.addWidget(self.cpu_chart)
        
        # Memory chart
        self.memory_chart = PerformanceChart('memory')
        charts_splitter.addWidget(self.memory_chart)
        
        layout.addWidget(charts_splitter)
        
        self.tabs.addTab(tab, "Performance Charts")
        
    def create_control_buttons(self, parent_layout):
        """Create control buttons"""
        button_layout = QHBoxLayout()
        
        self.pause_btn = QPushButton("⏸️ Pause Monitor")
        self.pause_btn.clicked.connect(self.toggle_monitoring)
        button_layout.addWidget(self.pause_btn)
        
        self.clear_history_btn = QPushButton("🗑️ Clear History")
        self.clear_history_btn.clicked.connect(self.clear_history)
        button_layout.addWidget(self.clear_history_btn)
        
        self.export_btn = QPushButton("💾 Export Report")
        self.export_btn.clicked.connect(self.export_performance_report)
        button_layout.addWidget(self.export_btn)
        
        parent_layout.addLayout(button_layout)
        
    def start_monitoring(self):
        """Start monitoring"""
        if not PSUTIL_AVAILABLE:
            # Update UI to show that monitoring is disabled
            self.cpu_label.setText("CPU: Monitoring disabled (psutil not available)")
            self.memory_label.setText("Memory: Monitoring disabled (psutil not available)")
            self.system_details_text.setText("System monitoring is disabled because psutil is not installed.\n"
                                            "To enable system monitoring, install psutil:\n"
                                            "pip install psutil")
            self.process_info_text.setText("Process monitoring is disabled (psutil not available)")
            return
            
        if self.system_monitor is None:
            self.system_monitor = SystemMonitorWorker()
            self.system_monitor.system_stats_updated.connect(self.update_system_stats)
            self.system_monitor.start()
            
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.system_monitor:
            self.system_monitor.stop()
            self.system_monitor = None
            
    def toggle_monitoring(self):
        """Toggle monitoring status"""
        if self.system_monitor and self.system_monitor.running:
            self.stop_monitoring()
            self.pause_btn.setText("▶️ Start Monitor")
        else:
            self.start_monitoring()
            self.pause_btn.setText("⏸️ Pause Monitor")
    
    def update_system_stats(self, stats):
        """Update system statistics"""
        if not stats:
            return
            
        try:
            # Save history data
            self.system_stats_history.append(stats)
            if len(self.system_stats_history) > 3600:  # Keep 1 hour of data
                self.system_stats_history.pop(0)
            
            # Update CPU information
            cpu_percent = stats['cpu']['percent']
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
            self.cpu_progress.setValue(int(cpu_percent))
            
            # Set CPU color warning
            if cpu_percent > 80:
                self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
            elif cpu_percent > 60:
                self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
            else:
                self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
            
            # Update memory information
            memory_percent = stats['memory']['percent']
            self.memory_label.setText(f"Memory: {memory_percent:.1f}%")
            self.memory_progress.setValue(int(memory_percent))
            
            # Set memory color warning
            if memory_percent > 80:
                self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
            elif memory_percent > 60:
                self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
            else:
                self.memory_progress.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")
            
            # Update detailed information
            details = f"""CPU Information:
  Cores: {stats['cpu']['count']}
  Frequency: {stats['cpu']['frequency']:.0f} MHz
  Usage: {cpu_percent:.1f}%

Memory Information:
  Total: {stats['memory']['total']:.2f} GB
  Used: {stats['memory']['used']:.2f} GB
  Available: {stats['memory']['available']:.2f} GB
  Usage: {memory_percent:.1f}%

Disk Information:
  Total Space: {stats['disk']['total']:.2f} GB
  Used: {stats['disk']['used']:.2f} GB
  Free Space: {stats['disk']['free']:.2f} GB
  Usage: {stats['disk']['percent']:.1f}%"""
            
            self.system_details_text.setText(details)
            
            # Update process information
            process_info = f"""Current Process Resource Usage:
Memory Usage: {stats['process']['memory_mb']:.2f} MB
CPU Usage: {stats['process']['cpu_percent']:.1f}%"""
            
            self.process_info_text.setText(process_info)
            
            # Update charts
            self.cpu_chart.update_data(cpu_percent)
            self.memory_chart.update_data(memory_percent)
            
            # Check performance alerts
            self.check_performance_alerts(stats)
            
        except Exception as e:
            print(f"Error updating system stats: {e}")
    
    def check_performance_alerts(self, stats):
        """Check performance alerts"""
        try:
            # High CPU warning - only show once per session
            if stats['cpu']['percent'] > 90:
                if not self.cpu_alert_sent and not self.cpu_alert_shown_once:
                    self.performance_alert.emit('cpu_high', f"High CPU usage: {stats['cpu']['percent']:.1f}%")
                    self.cpu_alert_sent = True
                    self.cpu_alert_shown_once = True  # Mark as shown, never show again
            else:
                self.cpu_alert_sent = False
            
            # High memory warning - only show once per session
            if stats['memory']['percent'] > 90:
                if not self.memory_alert_sent and not self.memory_alert_shown_once:
                    self.performance_alert.emit('memory_high', f"High memory usage: {stats['memory']['percent']:.1f}%")
                    self.memory_alert_sent = True
                    self.memory_alert_shown_once = True  # Mark as shown, never show again
            else:
                self.memory_alert_sent = False
            
            # Low disk space warning - keep as is for critical warnings
            if stats['disk']['percent'] > 90:
                self.performance_alert.emit('disk_full', f"Low disk space: {stats['disk']['percent']:.1f}%")
                
        except Exception as e:
            print(f"Error checking alerts: {e}")
    
    def start_task(self, task_id: str, task_name: str, total_steps: int = 100):
        """Start task"""
        self.task_tracker.start_task(task_id, task_name, total_steps)
        self.current_task_label.setText(f"Executing: {task_name}")
        self.current_task_progress.setMaximum(total_steps)
        self.current_task_progress.setValue(0)
        
    def update_task_progress(self, task_id: str, current_step: int, status_message: str = ""):
        """Update task progress"""
        self.task_tracker.update_progress(task_id, current_step, status_message)
        
        task_info = self.task_tracker.get_task_info(task_id)
        if task_info:
            self.current_task_progress.setValue(current_step)
            
            if status_message:
                self.task_status_label.setText(status_message)
            
            # Update time information
            elapsed = task_info['elapsed_time']
            estimated = task_info.get('estimated_time', 0)
            
            time_text = f"Elapsed: {elapsed:.1f}s"
            if estimated > 0:
                time_text += f" | Remaining: {estimated:.1f}s"
                
            self.time_info_label.setText(time_text)
    
    def complete_task(self, task_id: str, success: bool = True):
        """Complete task"""
        self.task_tracker.complete_task(task_id, success)
        
        if success:
            self.current_task_label.setText("Task Completed")
            self.task_status_label.setText("✅ Success")
        else:
            self.current_task_label.setText("Task Failed")
            self.task_status_label.setText("❌ Failed")
        
        # Update history table
        self.update_task_history_table()
    
    def update_task_history_table(self):
        """Update task history table"""
        try:
            tasks = self.task_tracker.get_all_tasks()
            
            self.task_history_table.setRowCount(len(tasks))
            
            for row, (task_id, task_info) in enumerate(tasks.items()):
                # Task name
                self.task_history_table.setItem(row, 0, QTableWidgetItem(task_info['name']))
                
                # Status
                status = task_info['status']
                status_item = QTableWidgetItem(status)
                if status == 'completed':
                    status_item.setBackground(QColor(200, 255, 200))
                elif status == 'failed':
                    status_item.setBackground(QColor(255, 200, 200))
                self.task_history_table.setItem(row, 1, status_item)
                
                # Progress
                progress = f"{task_info['current_step']}/{task_info['total_steps']}"
                self.task_history_table.setItem(row, 2, QTableWidgetItem(progress))
                
                # Duration
                elapsed = task_info.get('total_time', task_info['elapsed_time'])
                self.task_history_table.setItem(row, 3, QTableWidgetItem(f"{elapsed:.1f}s"))
                
                # Completion time
                if 'end_time' in task_info:
                    end_time = time.strftime("%H:%M:%S", time.localtime(task_info['end_time']))
                    self.task_history_table.setItem(row, 4, QTableWidgetItem(end_time))
                else:
                    self.task_history_table.setItem(row, 4, QTableWidgetItem("Running"))
                    
        except Exception as e:
            print(f"Error updating task history: {e}")
    
    def clear_history(self):
        """Clear history"""
        self.task_tracker = TaskProgressTracker()
        self.system_stats_history.clear()
        self.task_history_table.setRowCount(0)
        
    def export_performance_report(self):
        """Export performance report"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Performance Report", "performance_report.txt", "Text Files (*.txt)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("MatSci-ML Studio Performance Monitor Report\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # System information
                    if self.system_stats_history:
                        latest_stats = self.system_stats_history[-1]
                        f.write("System Information:\n")
                        f.write(f"CPU Cores: {latest_stats['cpu']['count']}\n")
                        f.write(f"CPU Usage: {latest_stats['cpu']['percent']:.1f}%\n")
                        f.write(f"Total Memory: {latest_stats['memory']['total']:.2f} GB\n")
                        f.write(f"Memory Usage: {latest_stats['memory']['percent']:.1f}%\n")
                        f.write(f"Disk Usage: {latest_stats['disk']['percent']:.1f}%\n\n")
                    
                    # Task history
                    f.write("Task Execution History:\n")
                    tasks = self.task_tracker.get_all_tasks()
                    for task_id, task_info in tasks.items():
                        f.write(f"Task: {task_info['name']}\n")
                        f.write(f"Status: {task_info['status']}\n")
                        f.write(f"Progress: {task_info['current_step']}/{task_info['total_steps']}\n")
                        elapsed = task_info.get('total_time', task_info['elapsed_time'])
                        f.write(f"Duration: {elapsed:.1f} seconds\n")
                        f.write("-" * 30 + "\n")
                        
                print(f"Performance report exported to: {file_path}")
                
        except Exception as e:
            print(f"Export report failed: {e}")
    
    def closeEvent(self, event):
        """Close event"""
        self.stop_monitoring()
        event.accept() 