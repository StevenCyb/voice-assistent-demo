from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from llama_cpp import llama_log_set
import ctypes

def my_log_callback(level, message, user_data):
    pass

log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(my_log_callback)
llama_log_set(log_callback, ctypes.c_void_p())

class Chat:
  def __init__(self) -> None:
    model_path = hf_hub_download(
      cache_dir="models",
      repo_id="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
      filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    )

    kwargs = {
      "model_path": model_path,
      "n_ctx": 4096,
      "max_tokens": 4096,
      "n_batch": 512, 
      "verbose": False,
    }
    llm = LlamaCpp(**kwargs)

    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant that responses to all questions directly and with no more than three sentences and not more than 400 words.<|eot_id|><|start_header_id|>user<|end_header_id|>
    {history}{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = PromptTemplate(input_variables=["history", "input"], template=prompt)

    memory = ConversationBufferWindowMemory(input_key="input", memory_key="history", k=10)
    memory.ai_prefix = "ASSISTANT"
    memory.human_prefix = "USER"

    self.conv_chain = ConversationChain(
      llm=llm,
      verbose=False,
      prompt=prompt,
      memory=memory
    )

  def ask(self, user_input: str) -> str:
    return self.conv_chain.predict(input=user_input)
  
if __name__ == "__main__":
  chat = Chat()
  print("User: My car is red.")
  print("Assistant:", chat.ask("My car is red."))
  print("User: What is the color of my car?")
  print("Assistant:", chat.ask("What is the color of my car?"))