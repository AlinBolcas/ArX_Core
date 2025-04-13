"""
minimalDemo.py

Ultra-minimal demo of the ArX agent system with adaptive workflow.
The orchestrator dynamically determines what steps are needed based on the task requirements.
Includes basic UI for input and media viewing capabilities.
"""

import os
import sys
import re
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from threading import Thread

# Add project root to path for imports - Fix path resolution
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent  # ArX_Core is just one level up from tests
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import necessary components
from src.III_agents.agentsGen import AgentGen
from src.VI_utils.utils import printColoured, quick_look
from src.I_integrations.replicate_API import ReplicateAPI

# Create the output directory if it doesn't exist
assets_dir = os.path.join(project_root, "data", "output", "assets")
os.makedirs(assets_dir, exist_ok=True)

# Initialize API instances
replicate_api = ReplicateAPI()

# Define the function that will run the workflow
def run_workflow(task_description, progress_callback=None, ui_result_callback=None):
    # ==== Minimal ArX Agent Workflow Demo ====
    printColoured(f"\n🚀 Running ArX Agent Workflow: '{task_description}' 🚀\n", "magenta")
    
    # Initialize AgentGen with integrated memory
    ag = AgentGen()  # Shared memory enabled by default
    
    if progress_callback:
        progress_callback(5, "Initializing agents...")
    
    # Create a pool of specialized agents with various capabilities
    ag.create_agent(
        name="Researcher", 
        system_prompt="You research information and provide insightful findings to help with tasks.", 
        log_color="green",
        tool_names=["web_crawl", "get_news", "get_current_datetime"]
    )
    
    ag.create_agent(
        name="Creator", 
        system_prompt="You create visual content based on descriptions and requirements.", 
        log_color="cyan",
        tool_names=["generate_image", "generate_music"]
    )
    
    ag.create_agent(
        name="Builder", 
        system_prompt="You build 3D models and technical implementations.", 
        log_color="blue",
        tool_names=["generate_threed"]
    )
    
    ag.create_agent(
        name="Orchestrator",
        system_prompt="""You are a workflow orchestrator that determines what steps are needed to complete a task.
Consider what information and resources are needed, and which specialist agents should be consulted.
IMPORTANT: Only use specialists when their specific skills are required by the task.
- Researcher: Use for tasks requiring information gathering or research
- Creator: Use for tasks requiring image generation or visual content
- Builder: Use for tasks requiring 3D modeling or technical implementation

CRITICAL WORKFLOW DEPENDENCIES:
1. If 3D modeling is required, you MUST first use the Creator agent to generate an actual image
2. Only after you have a real image URL from the Creator agent can you use the Builder agent for 3D modeling
3. Never use made-up or example URLs - only use actual generated image URLs

Think step-by-step about what is needed to fulfill the user's request completely.""",
        log_color="yellow"
    )
    
    if progress_callback:
        progress_callback(10, "Setting up workflow...")
    
    # Define available agents for orchestration
    agent_configs = [
        {"name": "Researcher", "description": "Researches information using web search and news."},
        {"name": "Creator", "description": "Creates visual content like images and music."},
        {"name": "Builder", "description": "Builds 3D models from images or descriptions."}
    ]
    
    # Create a task in shared memory
    task_id = ag.create_task(task_description)
    printColoured(f"📋 Task: {task_description}", "white")
    
    # Initialize workflow state
    workflow_state = {
        "task": task_description,
        "completed": False,
        "results": {},
        "steps_taken": [],
        "downloaded_files": []  # Track downloaded files
    }
    
    # Use a simple loop to let the orchestrator determine and execute the workflow
    max_iterations = 5  # Prevent infinite loops
    for iteration in range(max_iterations):
        progress_percent = 10 + (iteration * 15)  # Roughly distribute progress across iterations
        if progress_callback:
            progress_callback(progress_percent, f"Iteration {iteration+1}/{max_iterations}...")
        
        printColoured(f"\n=== ITERATION {iteration+1}/{max_iterations} ===", "magenta")
        
        # Create a summary of current workflow state
        state_summary = f"Task: {workflow_state['task']}\n\nSteps taken so far:\n"
        for i, step in enumerate(workflow_state["steps_taken"]):
            state_summary += f"{i+1}. {step['agent']} {step['action']}: {step['result'][:100]}...\n"
        
        # Let orchestrator decide next step
        orchestration_prompt = f"""
Current workflow state:
{state_summary}

Based on the current state:
1. Is the task completed? (yes/no)
2. If not, which specialist agent should handle the next step?
3. What exactly should they do?

Return your decision as a structured JSON with these fields:
- completed: boolean (true if task is complete)
- next_agent: string (name of the next agent to use, or null if completed)
- action: string (what the agent should do, or null if completed)
"""

        # Get orchestrator's decision
        decision = ag.get_agent("Orchestrator").structured_output(user_prompt=orchestration_prompt)
        
        # Safety checks for decision format
        if not isinstance(decision, dict):
            printColoured("⚠️ Orchestrator returned invalid decision format.", "red")
            decision = {"completed": False, "next_agent": "Researcher", "action": "Analyze the task requirements"}
        
        # Check if task is completed
        if decision.get("completed", False):
            printColoured("✅ Orchestrator determined the task is complete!", "green")
            workflow_state["completed"] = True
            break
            
        # Get the next agent and action
        next_agent_name = decision.get("next_agent")
        next_action = decision.get("action")
        
        if not next_agent_name or not next_action:
            printColoured("⚠️ Orchestrator provided incomplete next step. Ending workflow.", "red")
            break
            
        # Check for workflow dependency violations
        if next_agent_name == "Builder" and "3D" in next_action.lower() and "model" in next_action.lower():
            if not workflow_state.get("image_url"):
                printColoured("⚠️ Workflow dependency violation: Cannot generate 3D model without first creating an image.", "red")
                printColoured("Redirecting to Creator agent to generate an image first...", "yellow")
                next_agent_name = "Creator"
                next_action = f"Generate an image based on: {task_description}"
        
        printColoured(f"🔄 Orchestrator selected: {next_agent_name} to {next_action}", "yellow")
        
        # Execute the selected agent with appropriate tools
        next_agent = ag.get_agent(next_agent_name)
        if not next_agent:
            printColoured(f"❌ Agent '{next_agent_name}' not found", "red")
            break
            
        # Determine appropriate tools based on agent type
        tool_names = next_agent.config.tool_names
        
        # Prepare the action prompt - with special handling for Builder using image URL
        if next_agent_name == "Builder" and workflow_state.get("image_url") and ("3D" in next_action.lower() or "3d" in next_action.lower()):
            # Add the image URL to the prompt for the Builder agent
            image_url = workflow_state['image_url']
            
            # Debug logging for the Builder agent
            printColoured(f"🔍 DEBUG: Adding image URL to Builder agent: {image_url}", "blue")
            printColoured(f"🔍 DEBUG: URL format valid: {'https://' in image_url and ('.' in image_url)}", "blue")
            
            action_prompt = f"""Task: {task_description}

Current workflow state:
{state_summary}

Your assigned action: {next_action}

IMPORTANT: USE THIS EXACT IMAGE URL FOR 3D GENERATION: {image_url}

USE THE generate_threed TOOL with this image_url parameter.
"""
        else:
            action_prompt = f"Task: {task_description}\n\nCurrent workflow state:\n{state_summary}\n\nYour assigned action: {next_action}"
        
        # Execute the agent
        if progress_callback:
            progress_callback(progress_percent + 5, f"Running {next_agent_name} agent...")
        
        result = next_agent.chat_completion(
            user_prompt=action_prompt,
            max_tokens=300,
            tool_names=tool_names
        )
        
        # Save the response and update workflow state
        ag.save_agent_response(task_id, next_agent_name, action_prompt, result)
        
        # Print result
        printColoured(f"{next_agent_name}'s response:", "white")
        printColoured(result, next_agent.config.log_color)
        
        # Update workflow state
        workflow_state["results"][next_agent_name] = result
        workflow_state["steps_taken"].append({
            "agent": next_agent_name,
            "action": next_action,
            "result": result
        })
        
        # Extract image URL if present (for passing to 3D generation)
        if next_agent_name == "Creator" and ("image" in next_action.lower() or "visual" in next_action.lower()):
            import re
            # Look for URLs in markdown format or direct links
            image_url_match = re.search(r'\[.*?\]\((https?://[^\s]+\.(jpg|jpeg|png|gif|webp))\)', result)
            if not image_url_match:
                image_url_match = re.search(r'(https?://[^\s]+\.(jpg|jpeg|png|gif|webp))', result)
            
            if image_url_match:
                image_url = image_url_match.group(1)
                workflow_state["image_url"] = image_url
                printColoured(f"📸 Image URL extracted: {image_url}", "green")
                
                # Download and display the image
                if progress_callback:
                    progress_callback(progress_percent + 7, "Downloading image...")
                
                # Generate a unique filename based on timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_filename = f"image_{timestamp}.jpg"
                image_path = os.path.join(assets_dir, image_filename)
                
                try:
                    # Download the image using ReplicateAPI's download_file method
                    downloaded_path = replicate_api.download_file(
                        url=image_url,
                        output_dir=assets_dir,
                        filename=image_filename
                    )
                    
                    if downloaded_path:
                        printColoured(f"✅ Image downloaded to: {downloaded_path}", "green")
                        workflow_state["downloaded_files"].append({
                            "type": "image",
                            "url": image_url,
                            "path": downloaded_path
                        })
                        
                        # Display the image using quicklook
                        printColoured("🖼️ Opening image with QuickLook...", "cyan")
                        quick_look(downloaded_path)
                        
                        # Add a specific notification about the image being available for 3D modeling
                        if "3D" in task_description.lower() or "3d" in task_description.lower():
                            printColoured("✅ Image now available for 3D modeling", "green")
                    else:
                        printColoured("⚠️ Failed to download image.", "yellow")
                except Exception as e:
                    printColoured(f"⚠️ Error downloading image: {e}", "red")
            else:
                printColoured("⚠️ No valid image URL found in Creator's response.", "yellow")
        
        # Extract 3D model URL if present and download/display it
        if next_agent_name == "Builder" and any(term in next_action.lower() or term in task_description.lower() for term in ["3d", "model", "3-d"]):
            printColoured(f"🔍 DEBUG: Checking for 3D model URL in Builder's response...", "blue")
            printColoured(f"🔍 DEBUG: Response: {result[:100]}...", "blue")
            
            # Look for 3D model URLs in various formats
            # Try different patterns to catch various formatting
            # 1. Markdown link with various 3D extensions
            model_url_match = re.search(r'\[(.*?)\]\((https?://[^\s\)]+\.(glb|obj|fbx|usdz|stl|gltf))\)', result)
            if model_url_match:
                printColoured("🔍 DEBUG: Found 3D URL in markdown format", "blue")
            
            # 2. Direct URL with 3D extensions if markdown format not found
            if not model_url_match:
                model_url_match = re.search(r'(https?://[^\s\)]+\.(glb|obj|fbx|usdz|stl|gltf))', result)
                if model_url_match:
                    printColoured("🔍 DEBUG: Found 3D URL in direct URL format", "blue")
            
            # 3. Try a more generic URL pattern as last resort
            if not model_url_match:
                # Look for replicate.delivery URLs which might be 3D models
                model_url_match = re.search(r'(https?://replicate\.delivery/[^\s\)]+/output\.[a-zA-Z0-9]+)', result)
                if model_url_match:
                    printColoured("🔍 DEBUG: Found replicate.delivery URL that may be a 3D model", "blue")
            
            if model_url_match:
                model_url = model_url_match.group(1) if not model_url_match.group(2) else model_url_match.group(2)
                # Ensure we have the URL and not the link text
                if not model_url.startswith('http'):
                    model_url = model_url_match.group(2)
                
                printColoured(f"🧊 3D Model URL extracted: {model_url}", "green")
                
                # Download and display the 3D model
                if progress_callback:
                    progress_callback(progress_percent + 8, "Downloading 3D model...")
                
                # Generate a unique filename based on timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Try to get extension from URL or use .glb as fallback
                try:
                    model_ext = model_url.split('.')[-1]
                    # If extension isn't valid, use .glb
                    if model_ext not in ['glb', 'obj', 'fbx', 'usdz', 'stl', 'gltf']:
                        model_ext = 'glb'
                except:
                    model_ext = 'glb'
                    
                model_filename = f"model_{timestamp}.{model_ext}"
                model_path = os.path.join(assets_dir, model_filename)
                
                try:
                    printColoured(f"⬇️ Downloading 3D model from {model_url}", "blue")
                    # Download the 3D model using ReplicateAPI's download_file method
                    downloaded_path = replicate_api.download_file(
                        url=model_url,
                        output_dir=assets_dir,
                        filename=model_filename
                    )
                    
                    if downloaded_path:
                        printColoured(f"✅ 3D Model downloaded to: {downloaded_path}", "green")
                        workflow_state["downloaded_files"].append({
                            "type": "3d_model",
                            "url": model_url,
                            "path": downloaded_path
                        })
                        
                        # Display the 3D model using quicklook
                        printColoured("🧊 Opening 3D model with QuickLook...", "blue")
                        quick_look(downloaded_path)
                    else:
                        printColoured("⚠️ Failed to download 3D model.", "yellow")
                except Exception as e:
                    printColoured(f"⚠️ Error downloading 3D model: {e}", "red")
                    import traceback
                    printColoured(f"Stack trace: {traceback.format_exc()}", "red")
            else:
                printColoured("⚠️ No valid 3D model URL found in Builder's response.", "yellow")
                printColoured("Attempting to analyze Builder's raw response for URL text:", "yellow")
                
                # Print text fragments that look like they might contain URLs
                url_fragments = re.findall(r'https?://[^\s]+', result)
                if url_fragments:
                    printColoured(f"Found {len(url_fragments)} potential URL fragments:", "yellow")
                    for i, fragment in enumerate(url_fragments):
                        printColoured(f"Fragment {i+1}: {fragment}", "yellow")
    
    # Mark task as completed
    if workflow_state["completed"]:
        result_summary = "Task completed with the following steps:\n"
        for i, step in enumerate(workflow_state["steps_taken"]):
            result_summary += f"{i+1}. {step['agent']} {step['action']}\n"
    else:
        result_summary = "Task not fully completed. Reached maximum iterations."
    
    ag.update_task_status(task_id, "completed" if workflow_state["completed"] else "in_progress", result_summary)
    
    # Display the complete conversation history from memory
    printColoured("\n=== Complete Conversation History ===", "magenta")
    history = ag.get_conversation_as_string(task_id)
    print(history)
    
    # Display task status
    task_info = ag.get_task_info(task_id)
    printColoured(f"\nTask Status: {task_info['status']}", "green")
    printColoured(f"Result: {task_info['result']}", "green")
    
    # Display downloaded files summary
    if workflow_state["downloaded_files"]:
        printColoured("\n=== Downloaded Files ===", "magenta")
        for file_info in workflow_state["downloaded_files"]:
            printColoured(f"{file_info['type'].upper()}: {file_info['path']}", "cyan")
    
    printColoured("\n🏁 Demo Complete 🏁\n", "green")
    
    if progress_callback:
        progress_callback(100, "Completed!")
    
    # Send results back to UI
    if ui_result_callback:
        results = {
            "status": task_info['status'],
            "result": task_info['result'],
            "files": workflow_state["downloaded_files"],
            "history": history
        }
        ui_result_callback(results)

    return workflow_state

# Create simple UI
class MinimalUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ArX Minimal Demo")
        self.root.geometry("600x400")
        
        # Add a title
        self.title_label = tk.Label(root, text="ArX Agent Workflow Demo", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)
        
        # Description
        self.desc_label = tk.Label(root, text="Enter a task description and the agents will determine the workflow.")
        self.desc_label.pack(pady=5)
        
        # Task input frame
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.prompt_label = tk.Label(self.input_frame, text="Task Description:")
        self.prompt_label.pack(anchor=tk.W)
        
        self.prompt_entry = tk.Text(self.input_frame, height=3, width=50)
        self.prompt_entry.pack(fill=tk.X, pady=5)
        self.prompt_entry.insert(tk.END, "Design a futuristic smart chair that Apple and Meta would collaborate on, with integrated AR/VR capabilities and adaptive ergonomics")
        # Run button
        self.run_button = tk.Button(self.input_frame, text="Run Workflow", command=self.start_workflow)
        self.run_button.pack(pady=10)
        
        # Progress bar
        self.progress_frame = tk.Frame(root)
        self.progress_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        self.status_label = tk.Label(self.progress_frame, text="Ready")
        self.status_label.pack(anchor=tk.W, pady=5)
        
        # Results area
        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)
        
        self.results_label = tk.Label(self.results_frame, text="Results:")
        self.results_label.pack(anchor=tk.W)
        
        self.results_text = tk.Text(self.results_frame, height=10, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.config(state=tk.DISABLED)

    def update_progress(self, value, status_text):
        self.progress_bar["value"] = value
        self.status_label.config(text=status_text)
        self.root.update_idletasks()

    def update_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Format results nicely
        result_text = f"Status: {results['status']}\n\n"
        result_text += f"Result: {results['result']}\n\n"
        
        if results.get('files'):
            result_text += "Files:\n"
            for file in results['files']:
                result_text += f"- {file['type']}: {file['path']}\n"
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.config(state=tk.DISABLED)

    def start_workflow(self):
        task_description = self.prompt_entry.get("1.0", "end-1c").strip()
        if not task_description:
            self.status_label.config(text="Please enter a task description")
            return
        
        # Disable the button during execution
        self.run_button.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0
        self.status_label.config(text="Starting workflow...")
        
        # Clear previous results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Start workflow in a separate thread to keep UI responsive
        thread = Thread(target=self.run_workflow_thread, args=(task_description,))
        thread.daemon = True
        thread.start()

    def run_workflow_thread(self, task_description):
        try:
            # Run the workflow with progress updates
            run_workflow(task_description, self.update_progress, self.update_results)
        except Exception as e:
            # Update UI with error
            self.update_progress(0, f"Error: {e}")
        finally:
            # Re-enable the button
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))

# Main entry point - run UI or headless based on arguments
if __name__ == "__main__":
    # Check if running in UI mode or direct mode
    if len(sys.argv) > 1 and sys.argv[1] == "--no-ui":
        # Run in headless mode with a predefined task
        default_task = "Create a 3D character design of a realistic skunk for a TV commercial"
        run_workflow(default_task)
    else:
        # Run with UI
        root = tk.Tk()
        app = MinimalUI(root)
        root.mainloop()

