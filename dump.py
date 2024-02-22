from langchain_community.llms import LlamaCpp
""""

Inference with local model
"""
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
        model_path="C:/Users/jerem/Desktop/Doctilia/AI/Vigostral/models/vigostral-7b-chat.Q2_K.gguf",
        temperature=0.0,
        max_tokens=8,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
        n_ctx=1024
        #Verbose is required to pass to the callback manager
    )
    
   #output = self.model.invoke(prompt)