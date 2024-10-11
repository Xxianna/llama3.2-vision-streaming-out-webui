# llama3.2-vision-streaming-out-webui
借鉴 https://github.com/ThetaCursed/clean-ui.git 感觉这模型比较智障，欢迎交流参数

本文件采用了float16加载，由于使用了2080ti不支持bf16。如果均使用30以及之后显卡，可以改为原版的bf16

以及cuda设备的环境变量也可按需修改

模型文件请下载到本地，国内较快的地址 https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct

模型放在工作目录下的llama32vision11b文件夹下即可。环境为运行huggingface模型的常见环境


- This document uses float16 for loading, as the 2080ti does not support bf16. If you are using a 30 series or later graphics card, you can switch to the original bf16.
- Also, the CUDA device environment variables can be modified as needed.
- Please download the model file to your local drive. A faster address in China is available at https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct
- Place the model in the 'llama32vision11b' folder in the working directory.
- The python environment is the common environment for running Huggingface models.
- Feel free to share your thoughts on the model's parameters; I find this model somewhat lacking in intelligence.
