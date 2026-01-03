from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
import re
from dotenv import load_dotenv

load_dotenv()


loder = WebBaseLoader(["https://plato.stanford.edu/entries/heidegger/" ,"https://iep.utm.edu/heidegge/"])
documents = loder.load()

def remove_unnessesary_text(doc_text: str, start_text: str, end_text:str):
  start = re.escape(start_text)
  end = re.escape(end_text)
  pattern = re.compile(rf"{start}.*?{end}", re.DOTALL)
  new_text = re.sub(pattern, "", doc_text)
  return new_text

cleaned_text = remove_unnessesary_text(documents[1].page_content,"Heidegger, Martin | Internet Encyclopedia of Philosophy","Notes and Fragments" )

cleaned_text_doc1= remove_unnessesary_text(cleaned_text, 'Frühe Schriften (1912-16).','ISSN 2161-0002')

cleaned_text= remove_unnessesary_text(documents[0].page_content, 'Stanford Encyclopedia of Philosophy','Back to Top ')

cleaned_text_doc2= remove_unnessesary_text(cleaned_text, 'Kant und das Problem der Metaphysik,','ISSN 1095-5054')

documents[0].page_content=cleaned_text_doc2
documents[1].page_content=cleaned_text_doc1

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

db = Chroma.from_documents(chunks, embedding_model)

retriever = db.as_retriever(search_kwargs={"k": 4})

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Answer the question strictly using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

output_parser = StrOutputParser()



llm = ChatCohere(cohere_api_key='API_KEY')

final_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | output_parser
)



