SETUP STEPS--
1.i have first setuped environment
{python3 -m venv venv
venv\\Scripts\\activate }
2.then installed the required libraries
{langchain
faiss-cpu
python-dotenv
and so on}
3.made a env file and stored the api key
4.and saved another pdf file 'tsla-20231231.pdf'{this is a 10-k form file}
and the set up was ready

ANALIZE
load the pdf
PDF Loading: Used PyPDFLoader from langchain_community 
split the texts into chunks for making retrival search easy
Text Splitting:RecursiveCharacterTextSplitter to chunk text into 500-character segments with 50-character overlap.
then convert this text to embeddings
Embeddings:Generated vector embeddings with HuggingFace’s sentence-transformers/all-MiniLM-L6-v2 model.
then store the embedings
Vector Store: Stored embeddings in FAISS
then set up a retriver to give 2 chunks
Retrieval: retriever that returns top‑k=2 chunks
load the model
LLM Response: used Groq’s llama3-8b-8192
define a query .
send the query to retrevier and it will resultout 2 chunks
send that chunks and query to model it will result out answer

EXPERIMENT

Chunk size/overlap tuning: Tested various chunk_size and chunk_overlap values ( 400/50, 500/50, 600/100) to balance context fidelity
Temperature/Token limits: Adjusted LLaMA-3’s temperature (0.3 to 0.9) and max_tokens (200 to 500) to optimize answer clarity 

Final Comments
It can be extended with UI frameworks (e.g., Streamlit, Gradio) and chat history for conversational context.

REFERENCES
youtube playlist-compusx(playlist on langchain)
geeksforgeeks






