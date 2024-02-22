from qaVigostral import QA_inference
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from urllib.parse import unquote
from mistralai.client import MistralClient




from flask import Flask, request

app = Flask(__name__)


api_key = ""
model = "mistral-tiny"

model = MistralClient(api_key=api_key)


embedding = OpenAIEmbeddings(openai_api_key="")

db = FAISS.load_local("faiss", embedding)

retriever = db.as_retriever(search_kwargs={'k': 4, 'score_threshold': 0.9}, search_type="similarity")

qa = QA_inference(model = model, retriver= retriever)
print("loaded")

@app.route('/predict')
def predict():
    query = request.args.get('query')
    formated_query = unquote(query)
    output = qa.completion(formated_query, return_docs=False)
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 80)
    


   