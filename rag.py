# for making rag chatbot over a given pdf.i have chosen the pdf=ANNUAL REPORT ON FORM 10-K FOR THE YEAR ENDED DECEMBER 31, 2023.
# imported the required librarys
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import retrieval_qa
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
from openai import OpenAI

load_dotenv()  # this functions loads the api-keys from env file
groq_token = os.getenv("GROQ_API_KEY")

loader = PyPDFLoader("tsla-20231231.pdf")
pages = loader.load()  # this converts the pdf into lists
# this splits the whole list into tiny parts
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)  # stores that splitted documents
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # this embeddings converts the splitted documents to vector form
    cache_folder="./hf_cache")
# FAISS is vector store that stores this embedded vectors
vectorstore = FAISS.from_documents(chunks, embeddings)
# retriver takes the query and converts into vector form and searchs with semantic meaning in vectorstore
retriver = vectorstore.as_retriever(kwargs=2)
# this is the query we are going send to retrever
query = "what is teslas 2022 revenue"
# query is send to retrever by this invoke function
result = retriver.invoke(query)
# making llm model


def generate_answer_with_groq(query, context, groq_api_key):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key
    )
# making a dynamic promt which is going to be sent to model
    prompt = f"""You are an intelligent assistant. Use the following context to answer the question.
    Context:
    {context}
    Question:\n{query}"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content


# question should be entered in this 'query'part
query = "what is teslas 2023 revenue"
# query is been sent to retriever it converts it into vector and searches the relavent info
retrieved_docs = retriver.invoke(query)
# page content of retrived docs are joined (doc.page_content will generate pagecontent of each doc
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
# all these query and context is send to ai model and answer is given
final_answer = generate_answer_with_groq(query, context_text, groq_token)
print("Answer:", final_answer)
