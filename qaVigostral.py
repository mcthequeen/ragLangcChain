
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from mistralai.models.chat_completion import ChatMessage



class QA_inference:

  def __init__(self, model, retriver):
    self.model = model
    self.retriever = retriver

  def get_documents(self,query):
    results = self.retriever.get_relevant_documents(query)
    documents = ''
    for i in range(len(results)):
        documents += f"Document {i} \n " + str(results[i].page_content) + '\n'
    
    return documents, results

  def completion(self, query, return_docs = False):
    
    documents, results = self.get_documents(query)
    
    template = f"""Tu es un assistant qui doit répondre aux questions de l'utilisateur en te basant sur les documents.
Tu ne dois que te baser sur les documents et rien d'autre.
Tu dois répondre seulement en Français.
Si la réponse n'est pas dans les documents répond que tu ne peux pas répondre, et rien d'autre
Pour les mots techniques, mets une explication comme si l'utilisateur avait 5 ans.
Voici les documents:
{documents}
"""

        
    print("Iris :")
    messages = [
      ChatMessage(role="system", content=template),
      ChatMessage(role="user", content=query)
    ]

# With streaming
    stream_response = self.model.chat_stream(model="mistral-tiny", 
                                            temperature=0.0,
                                            max_tokens=512,
                                            top_p=1,
                                            messages=messages)

    output = []
    for chunk in stream_response:
      print(chunk.choices[0].delta.content, end="")
      output.append(chunk.choices[0].delta.content)


    if return_docs:
      for i in range(len(results)):
        print("\n======= Metada ======= \n")
        print(results[i].metadata)
        print("\n")
        print("======= Content ======= \n")
        print(results[i].page_content)
        
    return output