from qaVigostral import QA_inference
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from urllib.parse import unquote


from flask import Flask, request

app = Flask(__name__)


    


embedding = OpenAIEmbeddings(openai_api_key="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")

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

db = FAISS.load_local("faiss", embedding)

retriever = db.as_retriever(search_kwargs={'k': 2, 'score_threshold': 0.9}, search_type="similarity")

qa = QA_inference(model = llm, retriver= retriever)

print("loaded")

@app.route('/predict')
def predict():
    query = request.args.get('query')
    formated_query = unquote(query)
    output = qa.completion(formated_query, return_docs=False)
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 80)
    

   