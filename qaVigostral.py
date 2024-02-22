
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate




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
    print("model type: ",type(self.model))
    documents, results = self.get_documents(query)
    template = """Tu es un assistant qui doit répondre aux questions de l'utilisateur en te basant sur les documents.
Tu ne dois que te baser sur les documents et rien d'autre.
Tu dois répondre seulement en Français.
Si tu ne peux pas donner de réponse, dis le.
Pour les mots techniques, mets une explication comme si l'utilisateur avait 5 ans.
Voici les documents:
{documents}
Voici la question: {question}
"""
    prompt_template = PromptTemplate.from_template(template)

    prompt = prompt_template.format(documents=documents, 
                       question = query)


        
    print("Iris :")
    prompt
    output = self.model.invoke(prompt)

    if return_docs:
      for i in range(len(results)):
        print("\n======= Metada ======= \n")
        print(results[i].metadata)
        print("\n")
        print("======= Content ======= \n")
        print(results[i].page_content)
        
    return output