import gradio as gr
import torch
import os
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, GenerationConfig
import math
import threading

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or adjust size as needed
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"  #外接显卡优先
model_id = "llama32vision11b"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    device_map="auto",

    #2080Ti运行，雷电所以使用cuda:1，不支持bfloat16
    torch_dtype=torch.float16,
    # device_map="cuda:1:0",
)
processor = AutoProcessor.from_pretrained(model_id)

# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

oldpic = []

class my_streamer:
    def __init__(self):
        self.dist = ""

    def put(self, value):
        letter = processor.decode(value[0])
        self.dist = self.dist + letter

    def end(self):
        self.endend=True

    def set_image(self, image):
        global oldpic
        self.image = image
        # if(oldpic==[]):
        oldpic=image
        return "Load image successfully"

    def set_all(self, temperature, top_k, top_p, max_tokens, repetition_penalty=1.0):
        # self.image = image.value
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty


    # Function to process the image and generate a description
    # def describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens, history, repetition_penalty=1.0):
    def describe_image(self, message, history):
        global oldpic

        user_prompt = message
        self.image=oldpic
        image = self.image.resize(MAX_IMAGE_SIZE)

        # Prepare prompt with user input based
        prompt = f"<|image|><|begin_of_text|>{user_prompt} Answer:"
        # Preprocess the image and prompt
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        # Generate output with model

        self.dist = ""
        def model_thread():
            output = model.generate(
                **inputs,
                max_new_tokens=min(self.max_tokens, MAX_OUTPUT_TOKENS),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                # repetition_penalty=repetition_penalty,    #修改llama32vision11b\config.json的repetition_penalty=1.2，只修改文本模型
                streamer=self,
            )
        # Start the thread
        self.endend=False
        threading.Thread(target=model_thread).start()

        #线程没有结束时，不断返回
        while self.endend==False:
            yield self.dist
            #搜索是否存在Answer:，如果存在，则去掉其以及之前的部分
            # if "Answer:" in self.dist:
            #     cleaned_output = self.dist[self.dist.index("Answer:")+7:]
            #     break
            # else:
            #     cleaned_output = self.dist
            # yield cleaned_output

# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    Llama 3.2 vision 11b
    </h1>
    """)
        with gr.Row():
            # Left column with image and parameter inputs

            my_streamer_my = my_streamer()

            with gr.Column(scale=1):
                gr.Interface(my_streamer_my.set_image, gr.Image(
                    label="Image", 
                    type="pil", 
                    image_mode="RGB", 
                    height=512,  # Set the height
                    width=512   # Set the width
                ), "text")

                # Parameter sliders
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.6, step=0.1, interactive=True)
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=50, step=1, interactive=True)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50, interactive=True)
                repetition_penalty = gr.Slider(
                    label="重复惩罚", minimum=0.1, maximum=5.0, value=1.0, step=0.1, interactive=True)

            # Right column with the chat interface
            with gr.Column(scale=2):
                my_streamer_my.set_all(temperature.value, top_k.value, top_p.value, max_tokens.value, repetition_penalty.value)
                gr.ChatInterface(my_streamer_my.describe_image, type="messages" ,chatbot = gr.Chatbot(
                    label="Chatbot",
                    scale=1,
                    height=600,
                    type="messages",
                    autoscroll=True,
                ), textbox=gr.Textbox(
                    lines=5,
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    scale=7,
                    autofocus=True,
                    submit_btn=True,
                    stop_btn=True,
                ))

    return demo

# Launch the interface
demo = gradio_interface()
# demo.launch(share=True)   #创建共享链接，72小时有效

# ValueError: When localhost is not accessible, a shareable link must be created. Please set share=True or check your proxy settings to allow access to localhost.
# 对本地链接，如果报错，可以执行python.exe -m http.server，python将申请网络权限，从而打开对本软件的防火墙。可能有效，但还需要开头那句os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
demo.launch(share=False)