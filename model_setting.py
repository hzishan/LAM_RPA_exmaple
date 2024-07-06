from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from pathlib import Path
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage

def get_emodel():
    emodel_name="text2vec-large-chinese" # "GanymedeNil/text2vec-large-chinese" in HuggingFace
    emodel_path = Path.cwd() / Path(f'emodel/{emodel_name}')
    emodel_path = str(emodel_path)
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=emodel_path, model_kwargs={'device': 'cpu'})
        return embedding_model
    except FileNotFoundError:
        print("model not found in", emodel_path)
    except Exception as e:
        print(f"read File {emodel_path} got error: {e}")

model_list = {
    "breeze": "Breeze-7B-Instruct-v0.1-Q8_0",
    "taide": "taide-7b-a.2-q4_k_m",
    "taide2": "TAIDE-LX-7B-Chat.Q8_0",
    "mistral": "mistral-7b-instruct-v0.1.Q4_0",
    "llama": "Hermes-2-Pro-Llama-3-8B-Q8_0",
    3: "llama-2-7b.Q8_0",
    4: "llama-2-7b.Q5_K_M",
}

methods = {"ChatLlamaCpp":ChatLlamaCpp, 
           "LlamaCpp":LlamaCpp}

def get_llm(method="ChatLlamaCpp", model_name="llama", **kwargs):
    model = model_list[model_name]
    print("================== using model:", model, "==================")
    model_path = Path.cwd() / Path(f'model/{model}.gguf')
    try:
        llm = methods[method](
            model_path= str(model_path),
            n_gpu_layers=100,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            **kwargs
        )        
        return llm
    except FileNotFoundError:
        print("model not found in", model_path)
    except Exception as e:
        print(f"read File {model_path} got error: {e}")




if __name__ == "__main__":

    llm = get_llm(6)
    # llm = original_get_llm(6)
    messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Chinese. Translate the user sentence.",
    ),
    ("human", "I love programming."),
    ]
    result = llm.invoke(messages)
    print(result.content)

    

