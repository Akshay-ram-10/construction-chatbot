import gradio as gr
import os
import requests
import json
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

class ConstructionChatbot:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        
        self.system_prompt = """You are a specialized construction industry assistant. You only provide information, advice, and support related to:

- Building construction and architecture
- Construction materials and their properties
- Construction techniques and methods
- Building codes and regulations
- Safety protocols in construction
- Project management for construction projects
- Cost estimation and budgeting
- Structural engineering basics
- MEP (Mechanical, Electrical, Plumbing) systems
- Construction equipment and tools
- Sustainable and green building practices
- Quality control and inspections

If a user asks about topics unrelated to construction, politely redirect them back to construction-related topics. Always prioritize safety and compliance with local building codes in your responses."""

    def is_construction_related(self, message: str) -> bool:
        """Check if the message is construction-related"""
        construction_keywords = [
            'construction', 'building', 'concrete', 'steel', 'foundation', 'roofing',
            'plumbing', 'electrical', 'hvac', 'architecture', 'blueprint', 'contractor',
            'excavation', 'framing', 'drywall', 'flooring', 'insulation', 'windows',
            'doors', 'safety', 'permit', 'code', 'inspection', 'materials', 'tools',
            'equipment', 'project', 'budget', 'estimate', 'structural', 'mechanical',
            'residential', 'commercial', 'industrial', 'renovation', 'remodeling'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in construction_keywords)

    def get_response(self, message: str, history: List[Tuple[str, str]]) -> str:
        try:
            if not self.is_construction_related(message):
                return "I'm a specialized construction assistant. I can only help with construction-related topics such as building techniques, materials, safety, codes, project management, and other construction industry matters. Please ask me something related to construction!"
            
            messages = [{"role": "system", "content": self.system_prompt}]
            
            
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
            
            messages.append({"role": "user", "content": message})
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                return f"Sorry, I encountered an error: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "Sorry, the request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"Sorry, I encountered a network error: {str(e)}"
        except Exception as e:
            return f"Sorry, I encountered an unexpected error: {str(e)}"

def create_chatbot_interface():
    api_key = os.getenv("API_KEY")
    chatbot = ConstructionChatbot(api_key)    

    def chat_fn(message: str, history: List[Tuple[str, str]]):
        response = chatbot.get_response(message, history)
        history.append((message, response))
        return history, ""
    with gr.Blocks(
        title="Construction Industry Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; padding: 20px; }
        .chat-container { height: 500px; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="header">
            <h1>🏗️ Construction Industry Assistant</h1>
            <p>Ask me anything about construction, building materials, safety protocols, project management, and more!</p>
        </div>
        """)
        
        chatbot_ui = gr.Chatbot(
            value=[],
            elem_classes="chat-container",
            show_label=False,
            avatar_images=("🏗️", "🤖")
        )
        
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask about construction topics (e.g., 'What are the best materials for foundation?')",
                container=False,
                scale=4,
                show_label=False
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")
        
        gr.Examples(
            examples=[
                "What are the different types of concrete and their uses?",
                "How do I calculate the amount of steel required for a beam?",
                "What safety measures should be followed on a construction site?",
                "What are the latest building codes for residential construction?",
                "How do I estimate the cost of a small renovation project?",
                "What are the best practices for foundation waterproofing?",
                "How do I choose the right insulation material?",
                "What are the steps in the construction project lifecycle?"
            ],
            inputs=msg_box,
            examples_per_page=4
        )
        
        submit_btn.click(
            chat_fn,
            inputs=[msg_box, chatbot_ui],
            outputs=[chatbot_ui, msg_box]
        )
        
        msg_box.submit(
            chat_fn,
            inputs=[msg_box, chatbot_ui],
            outputs=[chatbot_ui, msg_box]
        )
        
        clear_btn = gr.Button("Clear Chat", variant="secondary")
        clear_btn.click(lambda: ([], ""), outputs=[chatbot_ui, msg_box])
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; color: #666;">
            <small>⚠️ This chatbot is specialized for construction-related queries only. 
            Always consult with licensed professionals for critical construction decisions.</small>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot_interface()
    demo.launch(
        share=True,  
        server_name="0.0.0.0", 
        server_port=7860
    )