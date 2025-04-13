import os
import sys
import json
import threading  # Import threading
import traceback  # Ensure traceback is imported
import logging  # Import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Callable  # Added Optional, Any, Callable
from pydantic import BaseModel, Field

# Configure logging FIRST - Replace all logging with no-ops
logging.basicConfig(level=logging.CRITICAL)

# Create dummy logger functions that do nothing
def noop(*args, **kwargs):
    pass

# Disable the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)
root_logger.info = noop
root_logger.warning = noop
root_logger.error = noop
root_logger.debug = noop
root_logger.critical = noop

# Disable all existing loggers
for name, logger in logging.Logger.manager.loggerDict.items():
    if isinstance(logger, logging.Logger):
        logger.setLevel(logging.CRITICAL)
        logger.info = noop
        logger.warning = noop
        logger.error = noop
        logger.debug = noop
        logger.critical = noop

# Set up module logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # Module default log level

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
# Navigate up to the project root (2 levels up from src/III_agents/)
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import TextGen and utils
from src.II_textGen.textGen import TextGen
from src.VI_utils.utils import printColoured  # Import printColoured
# Import AgentsMemory for direct integration
from src.III_agents.agentsMemory import AgentsMemory

# Disable all loggers from modules we import
for module in [
    'src.II_textGen.textGen', 'src.II_textGen.tools', 'src.II_textGen.rag', 
    'src.II_textGen.memory', 'faiss', 'src', 'src.I_integrations',
    'src.III_agents', 'src.III_agents.agentsGen'
]:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.CRITICAL)
    module_logger.info = noop
    module_logger.warning = noop
    module_logger.error = noop
    module_logger.debug = noop
    module_logger.critical = noop


# ---- Agent Configuration ----
class AgentConfig(BaseModel):
    """Stores configuration for a specific agent persona."""
    name: str
    system_prompt: Optional[str] = "You are a helpful assistant. Be concise."
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 100  # Default short max_tokens
    description: Optional[str] = "A general-purpose agent."
    log_color: Optional[str] = "white"  # Color for logging output
    context: Optional[str] = None  # Context documents for RAG
    system_context: Optional[str] = None  # System context documents for RAG
    tool_names: Optional[List[str]] = None  # Predefined tools the agent can use
    provider: Optional[str] = "openai"  # The LLM provider to use
    model: Optional[str] = None  # The specific model to use
    short_term_limit: Optional[int] = 8000  # Memory token limit


# Agent class representing an individual agent with its own TextGen instance
class Agent:
    """
    Individual agent with dedicated TextGen instance.
    Each agent has its own memory, tools, and configuration.
    """
    
    def __init__(
        self, 
        config: AgentConfig, 
        textgen_instance: Any = None,
        openai_api_key: Optional[str] = None,
        replicate_api_token: Optional[str] = None
    ):
        """
        Initialize an agent.
        
        Args:
            config: Configuration for this agent
            textgen_instance: Optional TextGen instance to use (shared)
            openai_api_key: Optional OpenAI API key
            replicate_api_token: Optional Replicate API token
        """
        self.config = config
        
        # Store an existing TextGen instance or create a dedicated one
        if textgen_instance:
            # Using a shared TextGen instance (like the orchestrator)
            self.textgen = textgen_instance
            self.has_shared_textgen = True
            printColoured(f"Agent '{config.name}' using shared TextGen instance. Note: Memory will be stored under the TextGen instance's agent name, not '{config.name}'.", config.log_color)
        else:
            # Import TextGen here to avoid circular imports
            from II_textGen.textGen import TextGen
            
            # Create a new TextGen instance with this agent's configuration
            self.textgen = TextGen(
                provider=config.provider,
                default_model=config.model,
                openai_api_key=openai_api_key,
                replicate_api_token=replicate_api_token,
                agent_name=config.name,  # Pass agent name for memory isolation
                short_term_limit=config.short_term_limit
            )
            self.has_shared_textgen = False
            printColoured(f"Agent '{config.name}' created with dedicated TextGen instance.", config.log_color)
            
    def chat_completion(self, user_prompt: str, system_prompt: str = None, 
                      temperature: float = None, max_tokens: int = None, 
                      context: str = None, system_context: str = None,
                      tool_names: List[str] = None, **kwargs) -> str:
        """
        Perform a chat completion using this agent's TextGen instance and configuration.
        """
        # Use agent's config values if not explicitly overridden
        final_system_prompt = system_prompt or self.config.system_prompt
        final_temperature = temperature if temperature is not None else self.config.temperature
        final_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        final_context = context or self.config.context
        final_system_context = system_context or self.config.system_context
        final_tool_names = tool_names or self.config.tool_names
        
        # Call the underlying TextGen instance with the agent's configuration
        return self.textgen.chat_completion(
            user_prompt=user_prompt,
            system_prompt=final_system_prompt,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            context=final_context,
            system_context=final_system_context,
            tool_names=final_tool_names,
            **kwargs
        )
    
    def structured_output(self, user_prompt: str, system_prompt: str = None,
                        temperature: float = None, max_tokens: int = None,
                        context: str = None, system_context: str = None, **kwargs) -> Any:
        """
        Get structured JSON output using this agent's TextGen instance and configuration.
        """
        # Use agent's config values if not explicitly overridden
        final_system_prompt = system_prompt or self.config.system_prompt
        final_temperature = temperature if temperature is not None else self.config.temperature
        final_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        final_context = context or self.config.context
        final_system_context = system_context or self.config.system_context
        
        # Call the underlying TextGen instance with the agent's configuration
        return self.textgen.structured_output(
            user_prompt=user_prompt,
            system_prompt=final_system_prompt,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            context=final_context,
            system_context=final_system_context,
            **kwargs
        )
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get the list of available tools for this agent."""
        return self.textgen.get_available_tools()
    
    def select_best_tools(self, user_prompt: str, top_k: int = 3) -> list:
        """
        Dynamically select the most relevant tools for the given prompt.
        
        Args:
            user_prompt: The user's request or task description
            top_k: Maximum number of tools to select
            
        Returns:
            List of selected tool names
        """
        # Get tools in a format that matches what we need
        available_tools = self.get_available_tools()
        available_tools_info = [f"- {tool['name']}: {tool['description']}"
                               for tool in available_tools]
        
        if not available_tools_info:
            printColoured(f"Agent '{self.config.name}': No tools available for selection.", self.config.log_color)
            return []

        # Use a structured prompt to get the exact format we need
        tool_selection_prompt = f"""
Task: "{user_prompt}"

Available tools:
{', '.join(available_tools_info)}

Select up to {top_k} tools that would be most useful for this task.
FORMAT INSTRUCTIONS: Return ONLY a JSON array of tool names as strings. No other text or explanation.
Example: ["tool1", "tool2"]
If no tools are useful, return empty array: []
"""

        try:
            selected_tools = self.structured_output(
                user_prompt=tool_selection_prompt,
                system_prompt="You are a tool selector that returns ONLY a JSON array of tool names. No other format is accepted.",
                store_interaction=False,
                max_tokens=100  # Increased slightly to handle longer array
            )

            # Handle various possible response formats consistently
            if isinstance(selected_tools, dict):
                # Case 1: {'tools': ['tool1', 'tool2']}
                if 'tools' in selected_tools and isinstance(selected_tools['tools'], list):
                    selected_tools = selected_tools['tools']
                # Case 2: Look for any other list values that might contain our tools
                else:
                    for key, value in selected_tools.items():
                        if isinstance(value, list) and all(isinstance(item, str) for item in value):
                            selected_tools = value
                            break
                    else:
                        # If we couldn't find a valid list, use an empty list
                        selected_tools = []

            # Ensure we have a proper list
            if not isinstance(selected_tools, list):
                selected_tools = []
            
            # Filter to only valid tool names
            valid_tool_names = [tool['name'] for tool in available_tools]
            final_selection = [name for name in selected_tools if isinstance(name, str) and name in valid_tool_names]
            
            if final_selection:  # Only log if tools were actually selected
                printColoured(f"Agent '{self.config.name}': Selected Tools for '{user_prompt[:30]}...': {final_selection}", self.config.log_color)
            
            return final_selection

        except Exception as e:
            printColoured(f"❌ Agent '{self.config.name}': Error during tool selection: {e}", "red")
            return []
    
    def clear_memory(self) -> None:
        """Clear this agent's memory."""
        self.textgen.clear_memory()
        
    def vision_analysis(self, 
                       image_url: str,
                       user_prompt: str,
                       system_prompt: str = None, 
                       system_context: str = None,
                       context: str = None,
                       temperature: float = None, 
                       max_tokens: int = None,
                       **kwargs) -> str:
        """
        Analyze an image with vision capabilities.
        
        Args:
            image_url: URL or path to the image to analyze
            user_prompt: The main user prompt
            system_prompt: Optional override for the agent's system prompt
            system_context: Optional override for the agent's system context
            context: Optional override for the agent's context
            temperature: Optional override for the agent's temperature
            max_tokens: Optional override for the agent's max tokens
            
        Returns:
            Generated response text
        """
        # Use agent's config values if not explicitly overridden
        final_system_prompt = system_prompt or self.config.system_prompt
        final_temperature = temperature if temperature is not None else self.config.temperature
        final_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        final_context = context or self.config.context
        final_system_context = system_context or self.config.system_context
        
        # Call the underlying TextGen instance with the agent's configuration
        return self.textgen.vision_analysis(
            image_url=image_url,
            user_prompt=user_prompt,
            system_prompt=final_system_prompt,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            context=final_context,
            system_context=final_system_context,
            **kwargs
        )
        
    def reasoned_completion(self,
                           user_prompt: str,
                           reasoning_effort: str = "low",
                           context: str = None,
                           max_tokens: int = None,
                           **kwargs) -> str:
        """
        Generate a reasoned completion.
        
        Args:
            user_prompt: The main user prompt
            reasoning_effort: Level of reasoning to apply ("low", "medium", "high")
            context: Optional override for the agent's context
            max_tokens: Optional override for the agent's max tokens
            
        Returns:
            Generated response text
        """
        # Use agent's config values if not explicitly overridden
        final_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        final_context = context or self.config.context
        
        # Call the underlying TextGen instance with the agent's configuration
        return self.textgen.reasoned_completion(
            user_prompt=user_prompt,
            reasoning_effort=reasoning_effort,
            context=final_context,
            max_tokens=final_max_tokens,
            **kwargs
        )
        
class AgentGen:
    """
    AgentGen: A multi-agent framework managing independent agents.
    Each agent has its own TextGen instance with isolated memory and tools.
    """

    def __init__(self,
                 provider: str = "openai",
                 openai_api_key: Optional[str] = None,
                 replicate_api_token: Optional[str] = None,
                 default_model: Optional[str] = None,
                 short_term_limit: int = 8000,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_shared_memory: bool = True,
                 memory_dir: Optional[str] = None):
        """
        Initialize the AgentGen system.
        
        Args:
            provider: Default provider for agents that don't specify one
            openai_api_key: OpenAI API key
            replicate_api_token: Replicate API token
            default_model: Default model for agents that don't specify one
            short_term_limit: Default memory token limit
            chunk_size: Default chunk size for RAG
            chunk_overlap: Default chunk overlap for RAG
            use_shared_memory: Whether to enable the shared memory system
            memory_dir: Optional custom directory for shared memory files
        """
        self.agents: Dict[str, Agent] = {}  # Store agents by name
        
        # Store configuration for creating new agents
        self.default_provider = provider
        self.openai_api_key = openai_api_key
        self.replicate_api_token = replicate_api_token
        self.default_model = default_model
        self.short_term_limit = short_term_limit
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create a orchestrator TextGen instance for general tasks and tool discovery
        self.orchestrator = TextGen(
            provider=provider,
            openai_api_key=openai_api_key,
            replicate_api_token=replicate_api_token,
            default_model=default_model,
            short_term_limit=short_term_limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            agent_name="orchestrator"  # Add agent_name for memory isolation
        )
        
        # Initialize shared memory system if enabled
        self.use_shared_memory = use_shared_memory
        if use_shared_memory:
            self.memory = AgentsMemory(memory_dir=memory_dir)
            printColoured(f"✅ Shared memory system enabled for agents.", "magenta")
        else:
            self.memory = None
            
        printColoured(f"🤖 AgentGen initialized with provider '{provider}' and model '{default_model}'.", "magenta")

    def create_agent(self,
                     name: str,
                     system_prompt: Optional[str] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     description: Optional[str] = None,
                     log_color: Optional[str] = "white",
                     context: Optional[str] = None,
                     system_context: Optional[str] = None,
                     tool_names: Optional[List[str]] = None,
                     provider: Optional[str] = None,
                     model: Optional[str] = None,
                     short_term_limit: Optional[int] = None,
                     shared_textgen: bool = False):
        """
        Creates and stores a new agent with its own TextGen instance.
        
        Args:
            name: Name of the agent
            system_prompt: Base system prompt for the agent
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens to generate
            description: Description of the agent's capabilities
            log_color: Color for this agent's logs ('red', 'green', 'blue', etc.)
            context: Context document for RAG
            system_context: System context document for RAG
            tool_names: Predefined tools the agent can use
            provider: Provider to use (defaults to AgentGen's provider)
            model: Model to use (defaults to provider's default)
            short_term_limit: Memory token limit (defaults to AgentGen's limit)
            shared_textgen: Whether to use the orchestrator's TextGen instance (default: False)
        """
        if name in self.agents:
            printColoured(f"⚠️ Agent '{name}' already exists. Overwriting.", log_color)
            
        # Validate tool names if provided
        valid_tool_names = tool_names
        if tool_names:
            available_tools = self.orchestrator.get_available_tools()
            available_tool_names = [tool['name'] for tool in available_tools]
            valid_tool_names = [name for name in tool_names if name in available_tool_names]
            
            if len(valid_tool_names) < len(tool_names):
                invalid_tools = [name for name in tool_names if name not in available_tool_names]
                printColoured(f"⚠️ Some tools for agent '{name}' are invalid and will be ignored: {invalid_tools}", log_color)
        
        # Create agent configuration
        config = AgentConfig(
            name=name,
            # Default to concise system prompt if none provided
            system_prompt=system_prompt or "You are a helpful assistant. Keep your response concise and under 100 tokens.",
            temperature=temperature,
            # Default to 100 tokens if not specified
            max_tokens=max_tokens if max_tokens is not None else 100, 
            description=description or f"Agent specializing in {name}.",
            log_color=log_color,
            context=context,
            system_context=system_context,
            tool_names=valid_tool_names,
            provider=provider or self.default_provider,
            model=model or self.default_model,
            short_term_limit=short_term_limit or self.short_term_limit
        )
        
        # Create the agent with its own TextGen instance or shared one
        if shared_textgen:
            agent = Agent(config, textgen_instance=self.orchestrator)
        else:
            agent = Agent(
                config, 
                openai_api_key=self.openai_api_key,
                replicate_api_token=self.replicate_api_token
            )
        
        # Store the agent
        self.agents[name] = agent
        
        # Print creation confirmation in the agent's color
        tool_info = f" with {len(valid_tool_names) if valid_tool_names else 0} TOOLS" if valid_tool_names else ""
        context_info = ", with CONTEXT." if context or system_context else ""
        model_info = f" using {config.provider}:{config.model or 'default'}" if not shared_textgen else " (shared TextGen)"
        printColoured(f"✅ Agent '{name}' created{tool_info}{context_info}{model_info} (Color: {log_color}).", log_color)
        
        return agent

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieves an agent by name."""
        return self.agents.get(name)

    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """Retrieves a stored agent configuration by name."""
        agent = self.agents.get(name)
        return agent.config if agent else None

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get all available tools from the orchestrator."""
        return self.orchestrator.get_available_tools()

    def select_best_tools(self, user_prompt: str, top_k: int = 3, agent_name: Optional[str] = None) -> list:
        """
        Dynamically selects the most relevant tools based on the user prompt.
        
        Args:
            user_prompt: The query to match tools against
            top_k: Maximum number of tools to return
            agent_name: Optional agent to use for tool selection
            
        Returns:
            List of selected tool names
        """
        # If agent name provided, use that agent's tool selection
        if agent_name:
            agent = self.get_agent(agent_name)
            if agent:
                return agent.select_best_tools(user_prompt, top_k)
        
        # Create a temporary agent using the orchestrator
        temp_agent = Agent(
            AgentConfig(name="temp", description="Temporary agent for tool selection"),
            textgen_instance=self.orchestrator
        )
        return temp_agent.select_best_tools(user_prompt, top_k)

    def loop_simple(self,
                    user_prompt: str,
                    agent_name: Optional[str] = None,
                    system_prompt: str = None,
                    temperature: float = None,
                    max_tokens: int = None,
                    context: str = None,
                    system_context: str = None,
                    tool_names: Optional[List[str]] = None,
                    use_dynamic_tools: bool = False,
                    max_depth: int = 3,
                    verbose: bool = True) -> str:
        """
        Iterative loop using tools, aiming for concise responses.
        
        Args:
            user_prompt: The main user prompt
            agent_name: Name of the agent to use (if any)
            system_prompt: Optional override for the agent's system prompt
            temperature: Optional override for the agent's temperature
            max_tokens: Optional override for the agent's max_tokens
            context: Optional override for the agent's context
            system_context: Optional override for the agent's system_context
            tool_names: Optional override for the agent's tool_names
            use_dynamic_tools: Whether to use dynamic tool selection (default: False)
            max_depth: Maximum number of iterations
            verbose: Whether to print verbose output
        """
        agent = None
        agent_log_color = "white"
        final_system_prompt = system_prompt
        final_temperature = temperature
        final_max_tokens = max_tokens if max_tokens is not None else 100
        final_context = context
        final_system_context = system_context
        final_tool_names = tool_names
        
        # Get the agent if specified
        if agent_name:
            agent = self.get_agent(agent_name)
            if agent:
                agent_log_color = agent.config.log_color
                if verbose: 
                    printColoured(f"⚙️ Using agent '{agent_name}' (Color: {agent_log_color}) in loop_simple.", "magenta")
                
                # Use agent's values unless explicitly overridden
                final_system_prompt = system_prompt or agent.config.system_prompt
                final_temperature = temperature if temperature is not None else agent.config.temperature
                final_max_tokens = max_tokens if max_tokens is not None else agent.config.max_tokens
                final_context = context or agent.config.context
                final_system_context = system_context or agent.config.system_context
                
                # Use provided tool_names, or agent's configured tools, or dynamic selection if requested
                if tool_names is not None:
                    final_tool_names = tool_names
                elif agent.config.tool_names and not use_dynamic_tools:
                    final_tool_names = agent.config.tool_names
                    if verbose: 
                        printColoured(f"🛠️ Using agent's predefined tools: {final_tool_names}", agent_log_color)
            else:
                if verbose: 
                    printColoured(f"⚠️ Agent '{agent_name}' not found. Using orchestrator.", "yellow")
        
        # Choose TextGen instance to use
        textgen = agent.textgen if agent else self.orchestrator
        
        response = ""
        agent_id = agent_name or 'Default'
        finished = False  # Flag to track if we've seen FINAL RESPONSE
        
        for i in range(max_depth):
            # Use dynamic tool selection only if requested and no predefined tools
            if use_dynamic_tools or final_tool_names is None:
                # Use the select_best_tools method with agent_name if we have one
                iteration_tools = self.select_best_tools(user_prompt, agent_name=agent_name if agent else None)
            else:
                iteration_tools = final_tool_names

            iterative_user_prompt = f"Task: {user_prompt}\n"
            if response: 
                 iterative_user_prompt += f"Current Answer: {response}\n"
            # Simplified prompt for brevity
            iterative_user_prompt += "Refine the answer concisely using tools if needed. If complete, state 'FINAL RESPONSE:'."

            response = textgen.chat_completion(
                user_prompt=iterative_user_prompt,
                system_prompt=final_system_prompt or "Be concise. Use tools if needed. State 'FINAL RESPONSE:' when done.",
                temperature=final_temperature,
                max_tokens=final_max_tokens,
                context=final_context,
                system_context=final_system_context,
                tool_names=iteration_tools
            )

            if verbose:
                printColoured(f"🔄 Simple Loop {i+1} [{agent_id}]: {response}", agent_log_color)
                
            if "FINAL RESPONSE:" in response:
                response = response.replace("FINAL RESPONSE:", "").strip()
                if verbose: 
                    printColoured(f"🏁 Simple Loop [{agent_id}] finished.", "green")
                finished = True
                break

        # Only show max depth warning if we didn't finish successfully
        if not finished and verbose:
            printColoured(f"⚠️ Simple Loop [{agent_id}] max depth ({max_depth}) reached.", "yellow")

        return response.replace("FINAL RESPONSE:", "").strip()

    def loop_react(self,
                   user_prompt: str,
                   agent_name: Optional[str] = None,
                   system_prompt: str = None,
                   temperature: float = None,
                   max_tokens: int = None,
                   context: str = None,
                   system_context: str = None,
                   tool_names: Optional[List[str]] = None,
                   use_dynamic_tools: bool = False,
                   max_depth: int = 3,
                   verbose: bool = True) -> str:
        """
        ReAct loop, aiming for concise steps.
        
        Args:
            user_prompt: The main user prompt
            agent_name: Name of the agent to use (if any)
            system_prompt: Optional override for the agent's system prompt
            temperature: Optional override for the agent's temperature
            max_tokens: Optional override for the agent's max_tokens
            context: Optional override for the agent's context
            system_context: Optional override for the agent's system_context
            tool_names: Optional override for the agent's tool_names
            use_dynamic_tools: Whether to use dynamic tool selection (default: False)
            max_depth: Maximum number of iterations
            verbose: Whether to print verbose output
        """
        agent = None
        agent_log_color = "white"
        final_system_prompt = system_prompt
        final_temperature = temperature
        final_max_tokens = max_tokens if max_tokens is not None else 100
        final_context = context
        final_system_context = system_context
        final_tool_names = tool_names
        agent_role = agent_name or 'Default'
        finished = False

        # Get the agent if specified
        if agent_name:
            agent = self.get_agent(agent_name)
            if agent:
                agent_log_color = agent.config.log_color
                if verbose: 
                    printColoured(f"⚙️ Using agent '{agent_name}' (Color: {agent_log_color}) in loop_react.", "magenta")
                
                # Use agent's values unless explicitly overridden
                final_system_prompt = system_prompt or agent.config.system_prompt
                final_temperature = temperature if temperature is not None else agent.config.temperature
                final_max_tokens = max_tokens if max_tokens is not None else agent.config.max_tokens
                final_context = context or agent.config.context
                final_system_context = system_context or agent.config.system_context
                
                # Use provided tool_names, or agent's configured tools, or dynamic selection if requested
                if tool_names is not None:
                    final_tool_names = tool_names
                elif agent.config.tool_names and not use_dynamic_tools:
                    final_tool_names = agent.config.tool_names
                    if verbose: 
                        printColoured(f"🛠️ Using agent's predefined tools: {final_tool_names}", agent_log_color)
            else:
                if verbose: 
                    printColoured(f"⚠️ Agent '{agent_name}' not found. Using orchestrator.", "yellow")
        
        # Choose TextGen instance to use
        textgen = agent.textgen if agent else self.orchestrator

        response = ""
        current_context = final_context or ""

        for i in range(max_depth):
            # Use dynamic tool selection only if requested and no predefined tools
            if use_dynamic_tools or final_tool_names is None:
                # Use shorter user prompt for tool selection if response exists
                tool_select_prompt = f"{user_prompt} Current state: {response}" if response else user_prompt
                # Use the select_best_tools method with agent_name if we have one  
                iteration_tools = self.select_best_tools(tool_select_prompt, agent_name=agent_name if agent else None)
            else:
                iteration_tools = final_tool_names

            # --- Step 1: OBSERVATION (Concise) ---
            obs_prompt = f"Task: {user_prompt}\n"
            if response: obs_prompt += f"Current State: {response}\n"
            obs_prompt += "Observe the current state briefly. What are the key facts?"

            observation = textgen.chat_completion(
                user_prompt=obs_prompt,
                system_prompt=final_system_prompt or "Observe concisely.",
                temperature=final_temperature, 
                max_tokens=final_max_tokens,
                context=current_context, 
                system_context=final_system_context,
                tool_names=iteration_tools
            )
            if verbose: 
                printColoured(f"🔄 React {i+1} [{agent_role}] OBSERVE: {observation}", agent_log_color)

            # --- Step 2: REFLECTION (Concise) ---
            refl_prompt = f"Task: {user_prompt}\nState: {response}\nObservation: {observation}\nReflect briefly: What is the next logical step?"

            reflection = textgen.chat_completion(
                user_prompt=refl_prompt,
                system_prompt=final_system_prompt or "Reflect concisely on the next step.",
                temperature=final_temperature, 
                max_tokens=final_max_tokens,
                context=current_context, 
                system_context=final_system_context,
                tool_names=iteration_tools
            )
            if verbose: 
                printColoured(f"🔄 React {i+1} [{agent_role}] REFLECT: {reflection}", agent_log_color)

            # --- Step 3: ACTION (Concise) ---
            # Improved action prompt to encourage complete sentences and proper termination
            act_prompt = f"""Task: {user_prompt}
Observation: {observation}
Plan: {reflection}

Execute the next step concisely using tools if needed. 
State 'FINAL RESPONSE:' only if task is complete.

IMPORTANT: Ensure your response is complete and doesn't cut off mid-sentence. 
If you're writing a story or creative content, craft complete sentences that fit within the token limit.
"""

            response = textgen.chat_completion(
                user_prompt=act_prompt,
                system_prompt=final_system_prompt or "Act concisely with complete sentences. Use tools if needed. State 'FINAL RESPONSE:' when done.",
                temperature=final_temperature, 
                max_tokens=final_max_tokens,
                context=current_context, 
                system_context=final_system_context,
                tool_names=iteration_tools
            )
            
            if verbose: 
                printColoured(f"🔄 React {i+1} [{agent_role}] ACTION: {response}", agent_log_color)

            # Check if response appears to be cut off mid-sentence
            if response.strip().endswith(('as if the', 'as if', 'such as', 'like a', 'like the')) or response.endswith(('...', '…')):
                if verbose: 
                    printColoured(f"⚠️ React response may be incomplete. Adding final period.", "yellow")
                response = response.rstrip('.…') + "."  # Ensure it ends properly
            
            if "FINAL RESPONSE:" in response:
                response = response.replace("FINAL RESPONSE:", "").strip()
                if verbose: 
                    printColoured(f"🏁 React Loop [{agent_role}] finished.", "green")
                finished = True
                break

        # Only show max depth warning if we didn't finish successfully
        if not finished and verbose:
            printColoured(f"⚠️ React Loop [{agent_role}] max depth ({max_depth}) reached.", "yellow")

        return response.replace("FINAL RESPONSE:", "").strip()

    def orchestrator_agent(self,
                     user_prompt: str,
                     handoff_agents: List[Dict[str, str]],
                     system_prompt: str = None,
                     temperature: float = 0.2,  # Lower default for more consistent decisions
                     max_tokens: int = 1000,
                     context: str = None,
                     system_context: str = None,
                     orchestration_instructions: str = None,
                     max_depth: int = 3,
                     verbose: bool = True) -> Union[List[str], str]:
        """
        Uses the orchestrator LLM to decide which agent(s) should handle a task.
        Analyzes the given task and available agents to determine the optimal routing.
        
        Args:
            user_prompt: Description of the task to route
            handoff_agents: List of available specialist agents with descriptions
            system_prompt: Optional override for orchestrator's system prompt
            temperature: Temperature setting for orchestrator (default: 0.2)
            max_tokens: Maximum tokens for orchestrator's response
            context: Optional additional context
            system_context: Optional system context
            orchestration_instructions: Optional specific instructions for agent selection
            max_depth: Maximum iterations for complex decisions
            verbose: Whether to print detailed orchestration logs
            
        Returns:
            List of agent names to handle the task, or error message string
        """
        if not handoff_agents:
            printColoured("⚠️ No handoff agents provided for orchestration.", "yellow")
            return "Error: No handoff agents specified."

        # Create a simplified list of agent names for the prompt
        agent_names = [agent['name'] for agent in handoff_agents]
        agent_options_text = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in handoff_agents])
        
        # Default orchestration instructions if none provided
        default_instructions = """
1. Select the best agent(s) for this request.
2. Return ONLY a JSON array of agent names as strings.
3. Example correct format: ["AgentName1", "AgentName2"]
4. If no agents fit, return: []
5. DO NOT use any other format or include any explanations.
"""
        
        selection_instructions = orchestration_instructions or default_instructions

        # Assemble the prompt with the task and available agents
        orchestration_prompt = f"""
User request: '{user_prompt}'

Available Agents:
{agent_options_text}

FORMAT INSTRUCTIONS:
{selection_instructions}

Available agent names: {', '.join(agent_names)}
"""

        # Use a specialized system prompt for orchestration
        default_system_prompt = "You are an orchestrator that routes tasks to specialists. Return ONLY a JSON array of agent names. Your entire response must be a valid JSON array of strings."
        orchestrator_system_prompt = system_prompt or default_system_prompt

        if verbose:
            printColoured(f"🚦 Orchestrator: Determining best agent for: '{user_prompt[:50]}...'", "magenta")

        try:
            # Use orchestrator's structured_output for consistent JSON responses
            selected_agent_names = self.orchestrator.structured_output(
                user_prompt=orchestration_prompt,
                system_prompt=orchestrator_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context,
                system_context=system_context
            )

            # Handle response format consistently - consolidate JSON parsing logic here
            if isinstance(selected_agent_names, dict):
                # Case 1: {'agents': ['Agent1', 'Agent2']}
                if 'agents' in selected_agent_names and isinstance(selected_agent_names['agents'], list):
                    selected_agent_names = selected_agent_names['agents']
                # Case 2: Look for agent names in dictionary keys that match our valid agent names
                else:
                    potential_agents = [k for k in selected_agent_names.keys() if k in agent_names]
                    if potential_agents:
                        selected_agent_names = potential_agents
                    # Case 3: Look for any list values that might contain agent names
                    else:
                        for key, value in selected_agent_names.items():
                            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                                selected_agent_names = value
                                break
                        else:
                            # If we couldn't find a valid list, use an empty list
                            selected_agent_names = []

            # Ensure we have a proper list
            if not isinstance(selected_agent_names, list):
                selected_agent_names = []
            
            # Filter to only valid agent names
            valid_agent_names = [agent['name'] for agent in handoff_agents]
            chosen_valid_names = [name for name in selected_agent_names if isinstance(name, str) and name in valid_agent_names]

            if verbose:
                if chosen_valid_names:
                    printColoured(f"🚦 Orchestrator Decision: Hand off to -> {', '.join(chosen_valid_names)}", "green")
                else:
                    printColoured("🚦 Orchestrator Decision: No specific agent selected.", "yellow")
            
            return chosen_valid_names

        except Exception as e:
            printColoured(f"❌ Orchestration Error: {e}", "red")
            return f"Error: Orchestration failed due to an exception: {e}"

    def get_agent_configs(self) -> Dict[str, AgentConfig]:
        """
        Returns a dictionary with all agent configurations.
        """
        return {name: agent.config for name, agent in self.agents.items()}
    
    def clear_all_memories(self) -> None:
        """
        Clear memory for all agents and the orchestrator.
        """
        self.orchestrator.clear_memory()
        for agent in self.agents.values():
            agent.clear_memory()
        
        # Also clear shared memory if enabled
        if self.use_shared_memory:
            self.clear_shared_memory()
            
        printColoured("Cleared memory for all agents, orchestrator, and shared memory.", "magenta")

    # ----- Shared Memory Integration Methods -----
    
    def create_task(self, task_description: str, initial_context: Optional[Dict] = None) -> str:
        """
        Create a new task in the shared memory system.
        
        Args:
            task_description: Description of the task
            initial_context: Optional initial context for the task
            
        Returns:
            task_id: Unique identifier for the task
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.create_task(task_description, initial_context)
    
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
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.save_agent_response(task_id, agent_name, input_message, output_message, metadata)
    
    def update_task_status(self, task_id: str, status: str, result: Optional[str] = None) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task
            status: New status ('in_progress', 'completed', 'failed')
            result: Optional final result for completed tasks
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        self.memory.update_task_status(task_id, status, result)
    
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
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.get_agent_response(task_id, agent_name, index)
    
    def get_all_agent_responses(self, task_id: str, agent_name: str) -> List[Dict]:
        """
        Get all responses from an agent for a specific task.
        
        Args:
            task_id: ID of the task
            agent_name: Name of the agent
            
        Returns:
            List of response data
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.get_all_agent_responses(task_id, agent_name)
    
    def get_conversation_history(self, task_id: str) -> List[Dict]:
        """
        Get the full conversation history for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of conversation entries in chronological order
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.get_conversation_history(task_id)
    
    def get_conversation_as_string(self, task_id: str) -> str:
        """
        Get the conversation history formatted as a string.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Formatted conversation history string
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.get_conversation_as_string(task_id)
    
    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """
        Get information about a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task data or None if not found
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.get_task_info(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict]:
        """
        Get information about all tasks.
        
        Returns:
            Dictionary of task data by task_id
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        return self.memory.get_all_tasks()
    
    def clear_shared_memory(self) -> None:
        """Clear all shared memory data."""
        if not self.use_shared_memory:
            raise ValueError("Shared memory is not enabled. Initialize AgentGen with use_shared_memory=True")
        self.memory.clear()
    
    # ----- End of Shared Memory Integration Methods -----


# ---- Worker Class ----
class Worker:
    """Runs an agent function in a separate thread."""
    def __init__(self, target_func: Callable, args: tuple = (), kwargs: dict = {}):
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs
        self.thread: Optional[threading.Thread] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        # Get agent color from kwargs if possible
        self.agent_name = kwargs.get('agent_name', 'Unknown')
        self.log_color = "grey"  # Default worker log color
        agent_gen_instance = kwargs.get('agent_gen_instance')  # Need instance to get color
        if agent_gen_instance and isinstance(agent_gen_instance, AgentGen):
            agent = agent_gen_instance.get_agent(self.agent_name)
            if agent: 
                self.log_color = agent.config.log_color

    def _run_target(self):
        try:
            self.kwargs['verbose'] = self.kwargs.get('verbose', True)
            # Note: Removing agent_gen_instance before calling target
            clean_kwargs = {k: v for k, v in self.kwargs.items() if k != 'agent_gen_instance'}
            printColoured(f"🧵 Worker starting [{self.agent_name}]...", self.log_color)
            self.result = self.target_func(*self.args, **clean_kwargs)
            printColoured(f"🧵 Worker finished [{self.agent_name}]. Result: {str(self.result)[:50]}...", self.log_color)
        except Exception as e:
            printColoured(f"🧵❌ Worker [{self.agent_name}] error: {e}", "red")
            self.error = e
            self.result = None
            # traceback.print_exc()

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.result = None
            self.error = None
            self.thread = threading.Thread(target=self._run_target)
            self.thread.daemon = True
            self.thread.start()
            # printColoured(f"🧵 Worker thread [{self.agent_name}] started.", self.log_color)
        else:
            printColoured(f"🧵 Worker [{self.agent_name}] already running.", self.log_color)

    def join(self, timeout: Optional[float] = None) -> None:
        if self.thread and self.thread.is_alive():
            # printColoured(f"🧵 Worker waiting for [{self.agent_name}]...", self.log_color)
            self.thread.join(timeout)
            if self.thread.is_alive():
                 printColoured(f"🧵⚠️ Worker [{self.agent_name}] timed out after {timeout}s.", "yellow")
            # else:
                 # printColoured(f"🧵 Worker thread [{self.agent_name}] finished.", self.log_color)
        # elif self.thread:
             # printColoured(f"🧵 Worker [{self.agent_name}] was already finished.", self.log_color)
        # else:
             # printColoured(f"🧵 Worker [{self.agent_name}] was never started.", "yellow")

    def get_result(self) -> Any:
        if self.error:
            printColoured(f"🧵 Result for [{self.agent_name}]: Error occurred -> {self.error}", "red")
            return f"Error in worker: {self.error}"
        if self.thread and not self.thread.is_alive():
             return self.result
        elif not self.thread:
             printColoured(f"🧵 Result for [{self.agent_name}]: Thread never started.", "yellow")
             return None
        else:  # Thread still running
             printColoured(f"🧵 Result for [{self.agent_name}]: Thread still running.", "yellow")
             return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 ArX Simplified Agent Orchestration Demo 🚀")
    print("="*60 + "\n")

    # Initialize AgentGen
    ag = AgentGen(provider="ollama")  # Defaults to openai
    
    # ---- 1. Register Simple Tools ----
    # Define a dummy function schema (doesn't actually call anything)
    dummy_tool_schema = {
        "type": "function",
        "function": {
            "name": "generate_idea",
            "description": "Generate a creative idea based on a theme",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "The theme to generate an idea for"
                    },
                    "complexity": {
                        "type": "integer",
                        "description": "How complex the idea should be (1-10)"
                    }
                },
                "required": ["theme"]
            }
        }
    }

    # ---- 2. Create Specialized Agents ----
    printColoured("\n--- Creating Specialized Agents ---", "magenta")
    
    # Creative agent
    ag.create_agent(
        name="CreativeExpert",
        system_prompt="You are a creative expert specializing in innovative ideas. Keep responses concise but imaginative.",
        max_tokens=100,
        description="Generates creative ideas for projects and concepts.",
        log_color="cyan"
    )
    
    # Technical agent
    ag.create_agent(
        name="TechnicalExpert",
        system_prompt="You are a technical expert specializing in practical implementation. Provide concise, actionable advice.",
        max_tokens=100,
        description="Provides technical expertise and implementation strategies.",
        log_color="blue"
    )
    
    # ---- 3. Create Orchestrator Agent ----
    ag.create_agent(
        name="Orchestrator",
        system_prompt="You are a decision-making orchestrator. Your role is to analyze requests and decide which specialist should handle them.",
        max_tokens=100,
        description="Routes tasks to appropriate specialized agents.",
        log_color="yellow"
    )
    
    # ---- 4. Define Agent Selection Logic ----
    def route_task(user_request):
        """Route a task to the appropriate agent based on orchestrator decision."""
        printColoured(f"\n🔄 Routing task: '{user_request}'", "magenta")
        
        # Get agent configs for orchestration
        agent_configs = [
            {"name": "CreativeExpert", "description": "Generates creative ideas for projects and concepts."},
            {"name": "TechnicalExpert", "description": "Provides technical expertise and implementation strategies."}
        ]
        
        # Use the improved orchestrator_agent function with task-specific instructions
        orchestration_instructions = """
        1. Carefully analyze the request to determine if it requires creative or technical expertise
        2. Select exactly ONE agent that is best suited to handle this request
        3. Return only a JSON array with a single agent name, example: ["TechnicalExpert"]
        """
        
        chosen_agents = ag.orchestrator_agent(
            user_prompt=user_request,
            handoff_agents=agent_configs,
            orchestration_instructions=orchestration_instructions,
            max_tokens=100,
            temperature=0.1  # Lower temperature for more consistent routing
        )
        
        if not chosen_agents:
            printColoured("⚠️ Orchestrator couldn't decide on an agent. Defaulting to CreativeExpert.", "yellow")
            return "CreativeExpert"
        
        selected_agent = chosen_agents[0]
        printColoured(f"✅ Orchestrator selected: {selected_agent}", "green")
        return selected_agent
    
    # ---- 5. Process Tasks ----
    # List of tasks to process
    tasks = [
        "Design a mascot character for a tech startup focused on AI.",
        "Explain the best way to implement a simple recommendation system.",
        "What time is it right now?"
    ]
    
    # Process each task
    for task in tasks:
        printColoured(f"\n{'='*40}", "white")
        printColoured(f"📝 NEW TASK: {task}", "white")
        
        # 1. Route the task
        selected_agent_name = route_task(task)
        selected_agent = ag.get_agent(selected_agent_name)
        
        # 2. Execute with simple loop
        if selected_agent:
            # Special case for time question - use a tool
            if "time" in task.lower():
                printColoured(f"🕒 Time query detected, using get_current_datetime tool", "white")
                tool_names = ["get_current_datetime"]
            else:
                tool_names = None
            
            # Execute task with the selected agent using simple loop
            result = ag.loop_simple(
                user_prompt=task,
                agent_name=selected_agent_name,
                max_depth=2,
                tool_names=tool_names
            )
            
            printColoured(f"📊 Result from {selected_agent_name}:", selected_agent.config.log_color)
            printColoured(result, selected_agent.config.log_color)
        else:
            printColoured(f"❌ Agent '{selected_agent_name}' not found", "red")
    
    print("\n" + "="*60)
    print("🏁 Demo Complete")
    print("="*60 + "\n")

