"""
agentsMemory.py

Shared memory system for multi-agent collaboration enabling message passing
between specialized agents while maintaining conversation history.
"""

import os
import sys
import json
import uuid
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities
from src.VI_utils.utils import printColoured

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class AgentsMemory:
    """
    Shared memory system for multi-agent collaboration that manages message passing
    between agents while preserving conversation history.
    
    This class provides a central memory store for:
    - Tracking conversation flow between multiple agents
    - Storing intermediate results and agent outputs
    - Maintaining task state across multiple agent interactions
    """
    
    def __init__(self, memory_dir: Optional[str] = None):
        """
        Initialize the shared memory system.
        
        Args:
            memory_dir: Optional custom directory for storing memory files
        """
        # Set up shared memory directory
        if memory_dir:
            self.memory_dir = Path(memory_dir)
        else:
            # Default to project's data/output/memory/shared directory
            project_root = Path(__file__).resolve().parent.parent.parent
            self.memory_dir = project_root / "data" / "output" / "memory" / "shared"
            
        # Create directory if it doesn't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory structures
        self.task_data = {}           # All data organized by task_id
        self.agent_outputs = {}       # Outputs from each agent by agent_name -> task_id
        self.conversation_history = {} # Sequential history by task_id
        
        # File paths for persistence
        self.task_file = self.memory_dir / "tasks.json"
        self.outputs_file = self.memory_dir / "agent_outputs.json"
        self.history_file = self.memory_dir / "conversation_history.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        printColoured("✅ AgentsMemory initialized for shared agent memory.", "green")
    
    def _initialize_files(self):
        """Initialize memory files if they don't exist."""
        if not self.task_file.exists():
            with open(self.task_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)
                
        if not self.outputs_file.exists():
            with open(self.outputs_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)
                
        if not self.history_file.exists():
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)
        
        # Load existing data if available
        self._load()
    
    def _load(self):
        """Load memory data from disk."""
        try:
            with open(self.task_file, "r", encoding="utf-8") as f:
                self.task_data = json.load(f)
                
            with open(self.outputs_file, "r", encoding="utf-8") as f:
                self.agent_outputs = json.load(f)
                
            with open(self.history_file, "r", encoding="utf-8") as f:
                self.conversation_history = json.load(f)
        except json.JSONDecodeError:
            # Handle case of corrupted files by starting fresh
            printColoured("⚠️ Memory files corrupted. Creating fresh memory.", "yellow")
            self.task_data = {}
            self.agent_outputs = {}
            self.conversation_history = {}
    
    def _save(self):
        """Save memory data to disk."""
        with open(self.task_file, "w", encoding="utf-8") as f:
            json.dump(self.task_data, f, indent=2)
            
        with open(self.outputs_file, "w", encoding="utf-8") as f:
            json.dump(self.agent_outputs, f, indent=2)
            
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def create_task(self, task_description: str, initial_context: Optional[Dict] = None) -> str:
        """
        Create a new task in the shared memory.
        
        Args:
            task_description: Description of the task
            initial_context: Optional initial context for the task
            
        Returns:
            task_id: Unique identifier for the task
        """
        task_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create task entry
        self.task_data[task_id] = {
            "task_description": task_description,
            "context": initial_context or {},
            "status": "created",
            "created_at": timestamp,
            "updated_at": timestamp
        }
        
        # Initialize conversation history for this task
        self.conversation_history[task_id] = [{
            "role": "system",
            "content": f"Task created: {task_description}",
            "timestamp": timestamp
        }]
        
        # Save to disk
        self._save()
        
        return task_id
    
    def save_agent_response(self, 
                           task_id: str,
                           agent_name: str,
                           input_message: str,
                           output_message: str,
                           metadata: Optional[Dict] = None) -> str:
        """
        Save a response from an agent to the shared memory.
        
        Args:
            task_id: ID of the task
            agent_name: Name of the agent
            input_message: Message received by the agent
            output_message: Response generated by the agent
            metadata: Optional additional information
            
        Returns:
            response_id: Unique identifier for the response
        """
        if task_id not in self.task_data:
            raise ValueError(f"Task {task_id} does not exist")
        
        timestamp = datetime.now().isoformat()
        response_id = str(uuid.uuid4())
        
        # Create the response entry
        response_entry = {
            "response_id": response_id,
            "agent_name": agent_name,
            "input_message": input_message,
            "output_message": output_message,
            "metadata": metadata or {},
            "timestamp": timestamp
        }
        
        # Update agent outputs
        if agent_name not in self.agent_outputs:
            self.agent_outputs[agent_name] = {}
        
        if task_id not in self.agent_outputs[agent_name]:
            self.agent_outputs[agent_name][task_id] = []
            
        self.agent_outputs[agent_name][task_id].append(response_entry)
        
        # Update conversation history
        self.conversation_history[task_id].append({
            "role": "user",
            "agent": agent_name,
            "content": input_message,
            "timestamp": timestamp
        })
        
        self.conversation_history[task_id].append({
            "role": "assistant",
            "agent": agent_name,
            "content": output_message,
            "timestamp": timestamp,
            "response_id": response_id
        })
        
        # Update task status and timestamp
        self.task_data[task_id]["updated_at"] = timestamp
        self.task_data[task_id]["status"] = "in_progress"
        
        # Save to disk
        self._save()
        
        return response_id
    
    def update_task_status(self, task_id: str, status: str, result: Optional[str] = None) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task
            status: New status ('in_progress', 'completed', 'failed')
            result: Optional final result for completed tasks
        """
        if task_id not in self.task_data:
            raise ValueError(f"Task {task_id} does not exist")
        
        timestamp = datetime.now().isoformat()
        
        # Update task data
        self.task_data[task_id]["status"] = status
        self.task_data[task_id]["updated_at"] = timestamp
        
        if result is not None:
            self.task_data[task_id]["result"] = result
            
        # Add to conversation history if completed
        if status in ["completed", "failed"]:
            self.conversation_history[task_id].append({
                "role": "system",
                "content": f"Task {status}: {result if result else ''}",
                "timestamp": timestamp
            })
        
        # Save to disk
        self._save()
    
    def get_agent_response(self, task_id: str, agent_name: str, index: int = -1) -> Optional[Dict]:
        """
        Get a specific response from an agent for a task.
        
        Args:
            task_id: ID of the task
            agent_name: Name of the agent
            index: Index of the response (-1 for most recent)
            
        Returns:
            Response data or None if not found
        """
        if (agent_name not in self.agent_outputs or 
            task_id not in self.agent_outputs.get(agent_name, {}) or
            not self.agent_outputs[agent_name][task_id]):
            return None
        
        responses = self.agent_outputs[agent_name][task_id]
        if not responses:
            return None
            
        # Handle negative indexing properly
        if index < 0:
            index = len(responses) + index
            
        if 0 <= index < len(responses):
            return responses[index]
        
        return None
    
    def get_all_agent_responses(self, task_id: str, agent_name: str) -> List[Dict]:
        """
        Get all responses from an agent for a specific task.
        
        Args:
            task_id: ID of the task
            agent_name: Name of the agent
            
        Returns:
            List of response data
        """
        if (agent_name not in self.agent_outputs or 
            task_id not in self.agent_outputs.get(agent_name, {})):
            return []
            
        return self.agent_outputs[agent_name][task_id]
    
    def get_conversation_history(self, task_id: str) -> List[Dict]:
        """
        Get the full conversation history for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of conversation entries in chronological order
        """
        if task_id not in self.conversation_history:
            return []
            
        return self.conversation_history[task_id]
    
    def get_conversation_as_string(self, task_id: str) -> str:
        """
        Get the conversation history formatted as a string.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Formatted conversation history string
        """
        if task_id not in self.conversation_history:
            return "No conversation history found."
            
        history = self.conversation_history[task_id]
        result = []
        
        for entry in history:
            role = entry["role"]
            if role == "system":
                result.append(f"SYSTEM: {entry['content']}")
            elif role == "user":
                result.append(f"{entry['agent']}: {entry['content']}")
            elif role == "assistant":
                result.append(f"RESPONSE FROM {entry['agent']}: {entry['content']}")
                
        return "\n\n".join(result)
    
    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """
        Get information about a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task data or None if not found
        """
        return self.task_data.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict]:
        """
        Get information about all tasks.
        
        Returns:
            Dictionary of task data by task_id
        """
        return self.task_data
    
    def clear(self) -> None:
        """Clear all memory data."""
        self.task_data = {}
        self.agent_outputs = {}
        self.conversation_history = {}
        self._save()
        printColoured("🧹 Cleared shared agent memory.", "yellow")

# Simple example usage
if __name__ == "__main__":
    memory = AgentsMemory()
    
    # Create a new task
    task_id = memory.create_task("Design a character concept and implement it in 3D")
    print(f"Created task: {task_id}")
    
    # Agent 1 creates a concept
    memory.save_agent_response(
        task_id=task_id,
        agent_name="CreativeExpert",
        input_message="Design a character concept for a tech startup mascot",
        output_message="I've designed 'Byte the Bot', a friendly robot with LED eyes and transparent body showing colorful circuits inside"
    )
    
    # Agent 2 implements the concept
    memory.save_agent_response(
        task_id=task_id,
        agent_name="TechnicalExpert",
        input_message="Implement the 'Byte the Bot' concept in 3D",
        output_message="I've created a 3D model with a translucent body shell, LED panel eyes, and internal circuit details visible through the shell"
    )
    
    # Get conversation history
    history = memory.get_conversation_as_string(task_id)
    print("\nConversation History:")
    print(history)
    
    # Get specific agent response
    creative_response = memory.get_agent_response(task_id, "CreativeExpert")
    print(f"\nCreative Response: {creative_response['output_message'] if creative_response else 'Not found'}")
    
    # Update task status
    memory.update_task_status(task_id, "completed", "Character concept created and implemented in 3D")
    
    # Get task info
    task_info = memory.get_task_info(task_id)
    print(f"\nTask Status: {task_info['status']}")
    print(f"Task Result: {task_info.get('result', 'N/A')}") 