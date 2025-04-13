"""
chat_textGen.py

Simple interactive chat interface using TextGen with customizable parameters.
Left panel: Chat interface
Right panel: Parameter controls
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from pathlib import Path
import time

# Add project root to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent  # ArX_Core is one level up
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import TextGen
from src.II_textGen.textGen import TextGen

class TextGenChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ArX TextGen Chat Interface")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Initialize TextGen with default values
        self.init_textgen()
        
        # Create main UI components
        self.create_ui()
        
        # Debug variables
        self.debug_mode = False
        self.messages = []
        
        # Log startup
        self.log_message("System", "Chat initialized. Select parameters on the right and start chatting!")
        
    def init_textgen(self, provider="openai", model=None):
        """Initialize or reinitialize the TextGen instance with current settings"""
        try:
            # Get default model based on provider
            if model is None:
                if provider == "openai":
                    model = "gpt-4o-mini"
                else:  # ollama
                    model = "gemma3:4b"
            
            # Initialize TextGen with selected provider
            self.textgen = TextGen(
                provider=provider,
                default_model=model,
                agent_name="ChatUI"  # Identify this instance for memory management
            )
            
            # Store current provider and model
            self.current_provider = provider
            self.current_model = model
            
            # Log initialization
            print(f"TextGen initialized with provider: {provider}, model: {model}")
            return True
        except Exception as e:
            print(f"Error initializing TextGen: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize TextGen: {e}")
            return False
    
    def create_ui(self):
        """Create the main UI components"""
        # Create main frame with 2 panels
        self.main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (chat interface)
        self.chat_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.chat_frame, weight=3)
        
        # Right panel (parameters)
        self.params_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.params_frame, weight=1)
        
        # Setup each panel
        self.setup_chat_panel()
        self.setup_params_panel()
    
    def setup_chat_panel(self):
        """Setup the chat interface panel (left side)"""
        # Create chat display area with larger font
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, 
                                                     width=60, height=30,
                                                     font=("Arial", 14))  # Increased font size
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # Add tag configurations for different roles with pastel colors
        self.chat_display.tag_config("user", foreground="#6495ED")  # Pale blue
        self.chat_display.tag_config("assistant", foreground="#E9967A")  # Pale orange
        self.chat_display.tag_config("system", foreground="gray")
        self.chat_display.tag_config("error", foreground="red")
        
        # Create input area
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Multi-line input text box with larger font
        self.user_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, 
                                                   width=40, height=4,
                                                   font=("Arial", 13))  # Increased font size
        self.user_input.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        # Bind both Enter and Ctrl+Return to send messages
        self.user_input.bind("<Control-Return>", self.send_message)
        self.user_input.bind("<Return>", self.send_message)
        
        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=5)
        
        # Send button
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.pack(fill=tk.X, expand=True, pady=5)
        
        # Clear button
        self.clear_button = ttk.Button(button_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(fill=tk.X, expand=True, pady=5)
    
    def setup_params_panel(self):
        """Setup the parameters panel (right side)"""
        # Create a notebook for tab organization
        self.params_notebook = ttk.Notebook(self.params_frame)
        self.params_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Basic settings tab
        self.basic_tab = ttk.Frame(self.params_notebook)
        self.params_notebook.add(self.basic_tab, text="Basic Settings")
        
        # Advanced settings tab
        self.advanced_tab = ttk.Frame(self.params_notebook)
        self.params_notebook.add(self.advanced_tab, text="Advanced")
        
        # Populate tabs
        self.setup_basic_settings()
        self.setup_advanced_settings()
    
    def setup_basic_settings(self):
        """Setup basic settings in the first tab"""
        # Main frame for basic settings with padding
        main_frame = ttk.Frame(self.basic_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create styles for ttk widgets
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 12))
        style.configure("Content.TLabel", font=("Arial", 11))
        style.configure("Content.TEntry", font=("Arial", 11))
        style.configure("Content.TCombobox", font=("Arial", 11))
        
        # Provider selection
        ttk.Label(main_frame, text="Provider:", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.provider_var = tk.StringVar(value="openai")
        provider_frame = ttk.Frame(main_frame)
        provider_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Radio buttons with command to apply settings immediately
        ttk.Radiobutton(provider_frame, text="OpenAI", variable=self.provider_var, 
                       value="openai", command=self.update_model_options).pack(side=tk.LEFT)
        ttk.Radiobutton(provider_frame, text="Ollama", variable=self.provider_var, 
                       value="ollama", command=self.update_model_options).pack(side=tk.LEFT)
        
        # Model selection
        ttk.Label(main_frame, text="Model:", style="Title.TLabel").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.model_menu = ttk.Combobox(main_frame, textvariable=self.model_var, width=25, style="Content.TCombobox")
        self.model_menu.grid(row=1, column=1, sticky=tk.W, pady=5)
        # Bind event to apply settings on selection
        self.model_menu.bind("<<ComboboxSelected>>", lambda e: self.apply_settings())
        
        # Output type selection
        ttk.Label(main_frame, text="Output Type:", style="Title.TLabel").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_type_var = tk.StringVar(value="chat_completion")
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Radio buttons with command to apply settings immediately
        ttk.Radiobutton(output_frame, text="Chat", variable=self.output_type_var, 
                       value="chat_completion", command=self.apply_settings).pack(anchor=tk.W)
        ttk.Radiobutton(output_frame, text="Structured", variable=self.output_type_var, 
                       value="structured_output", command=self.apply_settings).pack(anchor=tk.W)
        ttk.Radiobutton(output_frame, text="Reasoned", variable=self.output_type_var, 
                       value="reasoned_completion", command=self.apply_settings).pack(anchor=tk.W)
        
        # Temperature slider
        ttk.Label(main_frame, text="Temperature:", style="Title.TLabel").grid(row=3, column=0, sticky=tk.W, pady=5)
        temp_frame = ttk.Frame(main_frame)
        temp_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_slider = ttk.Scale(temp_frame, from_=0.0, to=1.0, length=150,
                                    variable=self.temp_var, orient=tk.HORIZONTAL)
        self.temp_slider.pack(side=tk.LEFT)
        # Bind event to apply settings on release
        self.temp_slider.bind("<ButtonRelease-1>", lambda e: self.apply_settings())
        
        temp_label = ttk.Label(temp_frame, textvariable=self.temp_var, width=5, style="Content.TLabel")
        temp_label.pack(side=tk.LEFT, padx=5)
        
        # Max tokens
        ttk.Label(main_frame, text="Max Tokens:", style="Title.TLabel").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.max_tokens_var = tk.IntVar(value=800)
        tokens_frame = ttk.Frame(main_frame)
        tokens_frame.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Note: ttk.Entry doesn't support style with font, will use standard tkinter Entry
        self.max_tokens_entry = tk.Entry(tokens_frame, textvariable=self.max_tokens_var, width=6, font=("Arial", 11))
        self.max_tokens_entry.pack(side=tk.LEFT)
        # Bind event to apply settings on focus out or Return
        self.max_tokens_entry.bind("<FocusOut>", lambda e: self.apply_settings())
        self.max_tokens_entry.bind("<Return>", lambda e: self.apply_settings())
        
        # System prompt
        ttk.Label(main_frame, text="System Prompt:", style="Title.TLabel").grid(row=5, column=0, sticky=tk.NW, pady=5)
        self.system_prompt_var = tk.StringVar(value="You are a helpful AI assistant.")
        # Note: scrolledtext.ScrolledText is a tkinter widget, not ttk, so font is fine
        self.system_prompt_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, 
                                                         width=30, height=4, font=("Arial", 11))
        self.system_prompt_text.grid(row=5, column=1, sticky=tk.W, pady=5)
        self.system_prompt_text.insert(tk.END, self.system_prompt_var.get())
        # Bind event to apply settings on focus out
        self.system_prompt_text.bind("<FocusOut>", lambda e: self.apply_settings())
        
        # NOW initialize model options after all variables are defined
        self.update_model_options(apply_immediately=False)
    
    def setup_advanced_settings(self):
        """Setup advanced settings in the second tab"""
        # Main frame for advanced settings with padding
        main_frame = ttk.Frame(self.advanced_tab, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create styles for the ttk widgets with larger fonts
        style = ttk.Style()
        style.configure("Large.TCheckbutton", font=("Arial", 12))
        style.configure("Large.TLabelframe.Label", font=("Arial", 12))
        style.configure("Large.TLabel", font=("Arial", 11))
        
        # Debug mode
        self.debug_var = tk.BooleanVar(value=False)
        debug_check = ttk.Checkbutton(main_frame, text="Debug Mode", 
                                   style="Large.TCheckbutton",
                                   variable=self.debug_var,
                                   command=self.toggle_debug)
        debug_check.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Memory management
        memory_frame = ttk.LabelFrame(main_frame, text="Memory Management", 
                                   style="Large.TLabelframe", padding=10)
        memory_frame.grid(row=1, column=0, sticky=tk.W+tk.E, pady=10)
        
        ttk.Button(memory_frame, text="Clear Memory", 
                  command=self.clear_memory).pack(fill=tk.X, padx=5, pady=5)
        
        # Status frame at the bottom
        status_frame = ttk.LabelFrame(main_frame, text="Status", 
                                   style="Large.TLabelframe", padding=10)
        status_frame.grid(row=2, column=0, sticky=tk.W+tk.E+tk.S, pady=10)
        
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_text, 
               style="Large.TLabel").pack(pady=5)
    
    def update_model_options(self, apply_immediately=True):
        """Update model dropdown options based on selected provider and apply settings"""
        provider = self.provider_var.get()
        
        if provider == "openai":
            models = [
                "gpt-4o", "gpt-4o-mini", 
                "gpt-4-turbo", "gpt-4",
                "gpt-3.5-turbo"
            ]
            default = "gpt-4o-mini"
        else:  # ollama
            models = [
                "llama3:8b", "llama3:70b",
                "gemma3:4b", "gemma3:9b",
                "mistral:7b", "mixtral:8x7b",
                "phi3:7b", "phi3:14b"
            ]
            default = "gemma3:4b"
        
        self.model_menu['values'] = models
        self.model_var.set(default)
        
        # Apply settings immediately when provider changes, but only if requested
        if apply_immediately:
            self.apply_settings()
    
    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = self.debug_var.get()
        message = "Debug mode enabled" if self.debug_mode else "Debug mode disabled"
        self.log_message("System", message)
    
    def clear_memory(self):
        """Clear TextGen memory"""
        try:
            self.textgen.clear_memory()
            self.log_message("System", "Memory cleared successfully")
        except Exception as e:
            self.log_message("Error", f"Failed to clear memory: {e}")
    
    def apply_settings(self):
        """Apply the current settings"""
        provider = self.provider_var.get()
        model = self.model_var.get()
        
        # Update system prompt from text widget only if it exists
        if hasattr(self, 'system_prompt_var') and hasattr(self, 'system_prompt_text'):
            self.system_prompt_var.set(self.system_prompt_text.get("1.0", tk.END).strip())
        
        # Check if provider or model changed
        if provider != self.current_provider or model != self.current_model:
            self.log_message("System", f"Switching to {provider} provider with model {model}...")
            
            # Reinitialize TextGen with new settings
            if self.init_textgen(provider, model):
                self.log_message("System", f"Now using {provider} with model {model}")
            else:
                self.log_message("Error", "Failed to apply model settings, using previous configuration")
                return
        
        # Log applied settings if in debug mode
        if self.debug_mode and hasattr(self, 'system_prompt_var'):
            settings = {
                "Provider": provider,
                "Model": model,
                "Output Type": self.output_type_var.get(),
                "Temperature": self.temp_var.get(),
                "Max Tokens": self.max_tokens_var.get(),
                "System Prompt": self.system_prompt_var.get()[:50] + "..." if len(self.system_prompt_var.get()) > 50 else self.system_prompt_var.get()
            }
            self.log_message("Debug", f"Applied settings: {json.dumps(settings, indent=2)}")
    
    def send_message(self, event=None):
        """Send user message and get AI response"""
        # Check if shift+return was pressed (multiline input)
        if event and event.keysym == 'Return' and not event.state & 0x1:  # No shift key
            user_message = self.user_input.get("1.0", tk.END).strip()
            if not user_message:
                return
                
            # Log user message
            self.log_message("User", user_message)
            
            # Clear input box
            self.user_input.delete("1.0", tk.END)
            
            # Disable send button to prevent multiple submissions
            self.send_button.config(state=tk.DISABLED)
            self.status_text.set("Processing...")
            
            # Process in a separate thread to keep UI responsive
            threading.Thread(target=self.process_message, args=(user_message,), daemon=True).start()
            
            # Prevent default handling of Return key (newline)
            return "break"
        
        # Allow default behavior for other key combinations
        return None
    
    def process_message(self, user_message):
        """Process user message and get AI response with streaming"""
        try:
            # Get current settings
            output_type = self.output_type_var.get()
            system_prompt = self.system_prompt_var.get()
            temperature = self.temp_var.get()
            max_tokens = self.max_tokens_var.get()
            
            # Debug log
            if self.debug_mode:
                self.log_message("Debug", f"Using output type: {output_type}, temp: {temperature}, max_tokens: {max_tokens}")
            
            # Set up timestamp for the assistant's message
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Start the assistant's message with timestamp
            self.root.after(0, lambda: self.start_assistant_message(timestamp))
            
            # Process based on output type
            if output_type == "chat_completion":
                # Use streaming for chat completions
                stream = self.textgen.chat_completion(
                    user_prompt=user_message,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True  # Always use streaming when available
                )
                
                # Process the stream in chunks
                full_response = ""
                for chunk in stream:
                    # Update UI with chunk
                    self.root.after(0, lambda c=chunk: self.update_stream_response(c))
                    full_response += chunk
                    
                    # Allow UI to update between chunks
                    time.sleep(0.01)
                
                # Store the interaction in memory
                self.textgen.memory.save_short_term(system_prompt, user_message, full_response)
                
            elif output_type == "structured_output":
                # Structured output doesn't support streaming
                response_obj = self.textgen.structured_output(
                    user_prompt=user_message,
                    system_prompt=system_prompt + " Return the output in structured JSON format.",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # Format JSON response
                response = json.dumps(response_obj, indent=2)
                self.root.after(0, lambda r=response: self.complete_assistant_response(r))
                
            elif output_type == "reasoned_completion":
                # Use streaming for reasoned completions too
                stream = self.textgen.reasoned_completion(
                    user_prompt=user_message,
                    system_prompt=system_prompt + " Show your reasoning step by step.",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True  # Always use streaming when available
                )
                
                # Process the stream
                full_response = ""
                for chunk in stream:
                    # Update UI with chunk
                    self.root.after(0, lambda c=chunk: self.update_stream_response(c))
                    full_response += chunk
                    
                    # Allow UI to update between chunks
                    time.sleep(0.01)
                
                # Store the interaction in memory
                self.textgen.memory.save_short_term(system_prompt, user_message, full_response)
                
            else:
                response = "Error: Unknown output type"
                self.root.after(0, lambda r=response: self.complete_assistant_response(r))
            
            # End the assistant's response
            self.root.after(0, self.end_assistant_response)
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}" if self.debug_mode else f"Error: {str(e)}"
            self.root.after(0, lambda: self.log_message("Error", error_msg))
        
        # Re-enable send button
        self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.status_text.set("Ready"))
    
    def start_assistant_message(self, timestamp):
        """Start a new assistant message in the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n[{timestamp}] AI: ", "assistant")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def update_stream_response(self, chunk):
        """Add a chunk of streamed response to the chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, chunk, "assistant")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def complete_assistant_response(self, response):
        """Add a complete (non-streamed) response to the chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, response, "assistant")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def end_assistant_response(self):
        """End the assistant's response with a newline"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n", "assistant")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def log_message(self, role, message):
        """Add a message to the chat display"""
        # Store message in history
        self.messages.append({"role": role, "content": message})
        
        # Update display
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format based on role
        if role == "User":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n", "user")
        elif role == "Assistant":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] AI: ", "assistant")
            self.chat_display.insert(tk.END, f"{message}\n", "assistant")
        elif role == "Error":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] ERROR: ", "error")
            self.chat_display.insert(tk.END, f"{message}\n", "error")
        else:  # System or Debug
            tag = role.lower()
            self.chat_display.insert(tk.END, f"\n[{timestamp}] {role}: ", tag)
            self.chat_display.insert(tk.END, f"{message}\n", tag)
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def clear_chat(self):
        """Clear the chat display but keep history in memory"""
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat display?"):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.log_message("System", "Chat cleared")

if __name__ == "__main__":
    # Create and run the app
    root = tk.Tk()
    app = TextGenChatApp(root)
    root.mainloop()
