import gradio as gr

def show_file(file):
    if file is None:
        return "No file uploaded"
    return f"Uploaded file: {file.name}"

with gr.Blocks() as demo:
    gr.Markdown("## 📂 Gradio File Upload Test")
    file_input = gr.File(label="Upload a file")  # single file upload
    output = gr.Textbox()
    file_input.change(fn=show_file, inputs=file_input, outputs=output)

demo.launch()


