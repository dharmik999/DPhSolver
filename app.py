import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ["HF_TOKEN"]
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

system_prompt = (
    "You are a strict physics problem solver. You will ONLY answer physics questions using correct physics formulas and principles. "
    "If someone asks about anything else (like shopping or restaurants), respond with: 'Sorry, I only solve physics problems.' "
    "Always explain step-by-step with units and clear logic. Always convert units to SI (kg, m, s) when needed. "
    "Do NOT guess. If the question lacks data, ask for clarification. "
    "Do NOT generate questions or continue the conversation unless explicitly prompted by the user."
)

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Only include the system prompt and current user message, ignoring history to prevent double conversation
    messages = [{"role": "system", "content": system_message or system_prompt}]
    messages.append({"role": "user", "content": message})

    try:
        # Use non-streaming response to avoid fragmented or unintended follow-ups
        response = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=False,  # Changed to False for complete response
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value=system_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.5, step=0.1, label="Temperature"),  # Lowered default for less randomness
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,  # Slightly lowered for more focused responses
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)