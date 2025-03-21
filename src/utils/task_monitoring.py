import time
import threading
import uuid
import weakref
from functools import wraps
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, Executor
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union, cast

from src.utils.enhanced_logging import get_enhanced_logger, monitoring_state
from src.utils.errors import TaskSchedulingError, FuturesError

# Create a logger for this module
logger = get_enhanced_logger('task_monitor')

# Type variable for return values
T = TypeVar('T')

class TaskState(Enum):
    """Enum for task states."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """Enum for task types."""
    PAGE_PROCESSING = "page_processing"
    EMBEDDING = "embedding"
    DATABASE = "database"
    API_CALL = "api_call"
    OTHER = "other"

class MonitoredTask:
    """Class for tracking a task's state and metadata."""
    
    def __init__(self, task_id: str, task_type: TaskType, description: str, url: Optional[str] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.url = url
        self.state = TaskState.PENDING
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[Exception] = None
        self.error_message: Optional[str] = None
        self.parent_task_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        
    def start(self) -> None:
        """Mark the task as started."""
        self.state = TaskState.RUNNING
        self.started_at = time.time()
        
    def succeed(self) -> None:
        """Mark the task as succeeded."""
        self.state = TaskState.SUCCEEDED
        self.completed_at = time.time()
        
    def fail(self, error: Exception) -> None:
        """
        Mark the task as failed.
        
        Args:
            error: The error that caused the failure
        """
        self.state = TaskState.FAILED
        self.completed_at = time.time()
        self.error = error
        self.error_message = str(error)
        
    def cancel(self) -> None:
        """Mark the task as cancelled."""
        self.state = TaskState.CANCELLED
        self.completed_at = time.time()
        
    def get_duration(self) -> Optional[float]:
        """
        Get the task duration in seconds.
        
        Returns:
            Optional[float]: Task duration in seconds, or None if not completed
        """
        if self.started_at is None:
            return None
            
        end_time = self.completed_at or time.time()
        return end_time - self.started_at
        
    def get_wait_time(self) -> Optional[float]:
        """
        Get the task wait time in seconds.
        
        Returns:
            Optional[float]: Wait time in seconds, or None if not started
        """
        if self.started_at is None:
            return None
            
        return self.started_at - self.created_at
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Returns:
            Dict[str, Any]: Task as a dictionary
        """
        result = {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.get_duration(),
            "wait_time": self.get_wait_time(),
            "metadata": self.metadata
        }
        
        if self.url:
            result["url"] = self.url
            
        if self.error_message:
            result["error_message"] = self.error_message
            
        if self.parent_task_id:
            result["parent_task_id"] = self.parent_task_id
            
        return result

class TaskRegistry:
    """Registry for tracking tasks."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskRegistry, cls).__new__(cls)
            cls._instance.tasks: Dict[str, MonitoredTask] = {}
            cls._instance.task_futures: Dict[str, weakref.ReferenceType] = {}
            cls._instance.lock = threading.Lock()
            cls._instance.stats = {
                "total_tasks": 0,
                "pending_tasks": 0,
                "running_tasks": 0,
                "succeeded_tasks": 0,
                "failed_tasks": 0,
                "cancelled_tasks": 0,
                "avg_duration": 0.0,
                "total_duration": 0.0,
                "tasks_by_type": {task_type.value: 0 for task_type in TaskType}
            }
        return cls._instance
    
    def create_task(self, task_type: TaskType, description: str, 
                   url: Optional[str] = None,
                   parent_task_id: Optional[str] = None) -> str:
        """
        Create a new task.
        
        Args:
            task_type: Type of the task
            description: Description of the task
            url: Optional URL being processed
            parent_task_id: Optional parent task ID
            
        Returns:
            str: ID of the created task
        """
        task_id = str(uuid.uuid4())
        task = MonitoredTask(task_id, task_type, description, url)
        
        if parent_task_id:
            task.parent_task_id = parent_task_id
            
        with self.lock:
            self.tasks[task_id] = task
            self.stats["total_tasks"] += 1
            self.stats["pending_tasks"] += 1
            self.stats["tasks_by_type"][task_type.value] += 1
            
        return task_id
        
    def start_task(self, task_id: str) -> None:
        """
        Mark a task as started.
        
        Args:
            task_id: ID of the task
        """
        with self.lock:
            if task_id not in self.tasks:
                return
                
            task = self.tasks[task_id]
            
            # Only transition from PENDING to RUNNING
            if task.state != TaskState.PENDING:
                return
                
            task.start()
            self.stats["pending_tasks"] -= 1
            self.stats["running_tasks"] += 1
        
    def complete_task(self, task_id: str, success: bool, error: Optional[Exception] = None) -> None:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the task
            success: Whether the task succeeded
            error: Optional error that occurred
        """
        with self.lock:
            if task_id not in self.tasks:
                return
                
            task = self.tasks[task_id]
            
            # Only transition from RUNNING to SUCCEEDED/FAILED
            if task.state != TaskState.RUNNING:
                return
                
            if success:
                task.succeed()
                self.stats["succeeded_tasks"] += 1
            else:
                task.fail(error or Exception("Unknown error"))
                self.stats["failed_tasks"] += 1
                
            self.stats["running_tasks"] -= 1
            
            # Update duration stats
            duration = task.get_duration()
            if duration:
                total_completed = self.stats["succeeded_tasks"] + self.stats["failed_tasks"]
                self.stats["total_duration"] += duration
                self.stats["avg_duration"] = self.stats["total_duration"] / total_completed
    
    def cancel_task(self, task_id: str) -> None:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task
        """
        with self.lock:
            if task_id not in self.tasks:
                return
                
            task = self.tasks[task_id]
            
            # Only cancel if not already completed
            if task.state in (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED):
                return
                
            # Update stats based on previous state
            if task.state == TaskState.PENDING:
                self.stats["pending_tasks"] -= 1
            elif task.state == TaskState.RUNNING:
                self.stats["running_tasks"] -= 1
                
            task.cancel()
            self.stats["cancelled_tasks"] += 1
            
            # Cancel the future if it exists
            future_ref = self.task_futures.get(task_id)
            if future_ref:
                future = future_ref()
                if future and not future.done():
                    future.cancel()
    
    def get_task(self, task_id: str) -> Optional[MonitoredTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Optional[MonitoredTask]: The task, or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_tasks_by_state(self, state: TaskState) -> List[MonitoredTask]:
        """
        Get tasks by state.
        
        Args:
            state: State to filter by
            
        Returns:
            List[MonitoredTask]: Tasks in the specified state
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.state == state]
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[MonitoredTask]:
        """
        Get tasks by type.
        
        Args:
            task_type: Type to filter by
            
        Returns:
            List[MonitoredTask]: Tasks of the specified type
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.task_type == task_type]
    
    def get_tasks_by_url(self, url: str) -> List[MonitoredTask]:
        """
        Get tasks by URL.
        
        Args:
            url: URL to filter by
            
        Returns:
            List[MonitoredTask]: Tasks for the specified URL
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.url == url]
    
    def associate_future(self, task_id: str, future: Future) -> None:
        """
        Associate a future with a task.
        
        Args:
            task_id: ID of the task
            future: Future to associate
        """
        with self.lock:
            # Store a weak reference to avoid memory leaks
            self.task_futures[task_id] = weakref.ref(future)
            
            # Add callbacks to update task state
            future.add_done_callback(lambda f: self._future_done_callback(task_id, f))
    
    def _future_done_callback(self, task_id: str, future: Future) -> None:
        """
        Callback for when a future completes.
        
        Args:
            task_id: ID of the task
            future: The completed future
        """
        # Check if cancelled
        if future.cancelled():
            self.cancel_task(task_id)
            return
            
        try:
            # Get the result to check for exceptions
            future.result()
            self.complete_task(task_id, True)
        except Exception as e:
            self.complete_task(task_id, False, e)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get task statistics.
        
        Returns:
            Dict[str, Any]: Task statistics
        """
        with self.lock:
            return self.stats.copy()
    
    def clean_old_tasks(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clean up old completed tasks.
        
        Args:
            max_age_seconds: Maximum age in seconds for completed tasks
            
        Returns:
            int: Number of tasks cleaned up
        """
        now = time.time()
        to_remove = []
        
        with self.lock:
            for task_id, task in self.tasks.items():
                if task.state in (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED):
                    if task.completed_at and (now - task.completed_at) > max_age_seconds:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
                if task_id in self.task_futures:
                    del self.task_futures[task_id]
                    
        return len(to_remove)

# Create the global task registry
task_registry = TaskRegistry()

# Decorators for monitoring functions
def monitored_task(task_type: TaskType, description_template: str):
    """
    Decorator to monitor a function as a task.
    
    Args:
        task_type: Type of the task
        description_template: Template for the task description
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Format the description with kwargs
            description = description_template
            for key, value in kwargs.items():
                description = description.replace(f"{{{key}}}", str(value))
                
            # Create a task
            url = kwargs.get('url', None)
            parent_task_id = kwargs.get('parent_task_id', None)
            
            task_id = task_registry.create_task(
                task_type, 
                description, 
                url=url,
                parent_task_id=parent_task_id
            )
            
            # Start the task
            task_registry.start_task(task_id)
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Complete the task
                task_registry.complete_task(task_id, True)
                
                return result
            except Exception as e:
                # Complete the task with error
                task_registry.complete_task(task_id, False, e)
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator

# Monkey-patching for thread and process pool executors
def monitor_executor(executor: Executor, name: str) -> Executor:
    """
    Monkey-patch an executor to monitor its tasks.
    
    Args:
        executor: The executor to monitor
        name: Name for the executor
        
    Returns:
        Executor: The monitored executor
    """
    # Store the original submit method
    original_submit = executor.submit
    
    # Define a new submit method that monitors the submitted tasks
    def monitored_submit(fn, *args, **kwargs):
        # Extract task metadata from kwargs
        task_type = kwargs.pop('task_type', TaskType.OTHER)
        description = kwargs.pop('description', f"Task in {name}")
        url = kwargs.pop('url', None)
        parent_task_id = kwargs.pop('parent_task_id', None)
        
        # Create a task
        task_id = task_registry.create_task(
            task_type, 
            description, 
            url=url,
            parent_task_id=parent_task_id
        )
        
        # Wrap the function to update task state
        def monitored_fn(*inner_args, **inner_kwargs):
            # Start the task
            task_registry.start_task(task_id)
            
            try:
                # Execute the original function
                result = fn(*inner_args, **inner_kwargs)
                
                # Complete the task
                task_registry.complete_task(task_id, True)
                
                return result
            except Exception as e:
                # Complete the task with error
                task_registry.complete_task(task_id, False, e)
                
                # Re-raise the exception
                raise
        
        # Submit the wrapped function
        future = original_submit(monitored_fn, *args, **kwargs)
        
        # Associate the future with the task
        task_registry.associate_future(task_id, future)
        
        return future
    
    # Replace the submit method
    executor.submit = monitored_submit  # type: ignore
    
    return executor

# Functions for working with monitored thread pools
def create_monitored_thread_pool(max_workers: Optional[int] = None, name: str = "ThreadPool") -> ThreadPoolExecutor:
    """
    Create a monitored thread pool.
    
    Args:
        max_workers: Maximum number of worker threads
        name: Name for the thread pool
        
    Returns:
        ThreadPoolExecutor: The monitored thread pool
    """
    executor = ThreadPoolExecutor(max_workers=max_workers)
    return monitor_executor(executor, name)

def create_monitored_process_pool(max_workers: Optional[int] = None, name: str = "ProcessPool") -> ProcessPoolExecutor:
    """
    Create a monitored process pool.
    
    Args:
        max_workers: Maximum number of worker processes
        name: Name for the process pool
        
    Returns:
        ProcessPoolExecutor: The monitored process pool
    """
    executor = ProcessPoolExecutor(max_workers=max_workers)
    return monitor_executor(executor, name)

# Utility functions for working with the task registry
def get_task_stats() -> Dict[str, Any]:
    """
    Get task statistics.
    
    Returns:
        Dict[str, Any]: Task statistics
    """
    return task_registry.get_stats()

def get_active_tasks() -> List[Dict[str, Any]]:
    """
    Get active tasks.
    
    Returns:
        List[Dict[str, Any]]: Active tasks
    """
    running_tasks = task_registry.get_tasks_by_state(TaskState.RUNNING)
    pending_tasks = task_registry.get_tasks_by_state(TaskState.PENDING)
    
    return [task.to_dict() for task in running_tasks + pending_tasks]

def get_failed_tasks() -> List[Dict[str, Any]]:
    """
    Get failed tasks.
    
    Returns:
        List[Dict[str, Any]]: Failed tasks
    """
    failed_tasks = task_registry.get_tasks_by_state(TaskState.FAILED)
    return [task.to_dict() for task in failed_tasks]

def cancel_all_tasks() -> int:
    """
    Cancel all active tasks.
    
    Returns:
        int: Number of tasks cancelled
    """
    running_tasks = task_registry.get_tasks_by_state(TaskState.RUNNING)
    pending_tasks = task_registry.get_tasks_by_state(TaskState.PENDING)
    
    count = 0
    for task in running_tasks + pending_tasks:
        task_registry.cancel_task(task.task_id)
        count += 1
        
    return count

def clean_up_old_tasks(max_age_seconds: float = 3600.0) -> int:
    """
    Clean up old completed tasks.
    
    Args:
        max_age_seconds: Maximum age in seconds for completed tasks
        
    Returns:
        int: Number of tasks cleaned up
    """
    return task_registry.clean_old_tasks(max_age_seconds)

def record_task_success(task_type_str: str, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Simple utility to record a successful task without the need to create and update it separately.
    
    Args:
        task_type_str: String representation of the task type
        description: Description of the task
        metadata: Optional metadata for the task
        
    Returns:
        str: The ID of the created task
    """
    # Convert string task type to enum
    try:
        task_type = TaskType[task_type_str.upper()] if isinstance(task_type_str, str) else task_type_str
    except (KeyError, AttributeError):
        task_type = TaskType.OTHER
        
    # Generate description if none provided
    if not description:
        description = f"Completed {task_type_str} operation"
        
    # Add metadata to description if provided
    if metadata:
        metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items() if v is not None)
        if metadata_str:
            description = f"{description} ({metadata_str})"
            
    # Get registry instance
    registry = TaskRegistry()
    
    # Create and immediately complete the task
    task_id = registry.create_task(task_type, description)
    registry.start_task(task_id)
    registry.complete_task(task_id, True)
    
    return task_id 