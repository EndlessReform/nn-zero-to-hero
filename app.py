import gradio
import gpt

# Intended for HuggingFace Spaces for GPT
config = gpt.GPTConfig("gpt_config.json")
model = gpt.GPT(
    vocab_size=config.vocab_size,
    context_size=config.context_size,
    embedding_dim=config.embedding_dim
)

# Use GPU if enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load checkpoint
model, _, _ = load_checkpoint(model, None, config.restore_path)

demo = gr.Interface(
    fn=lambda *args: generate(model, config, *args),
    inputs=[
        gr.Textbox(lines=2, placeholder="Prompt here..."),
        gr.Number(precision=0, value=256),
        gr.Number(value=0.8),
        gr.Slider(maximum=128, value=10),
        gr.Slider(maximum=1, value=1)
    ],
    outputs="text",
    title="Shakespeare-GPT",
    description="Putting theater kids out of their nonexistent jobs since 2023"
)

demo.launch()