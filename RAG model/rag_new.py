import ollama
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


path=path = r"C:\Users\rsaba\Downloads\fastapi-demo-products-get-post1\simplified-machine-learning-dr-pooja-sharma.pdf"

with open (path,'rb') as f:
    reader=PyPDF2.PdfReader(f)
    text=""
    for page in reader.pages:
        text+=page.extract_text()

#print(text[:100])

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunk=text_splitter.split_text(text)

# print(len(chunk))
# print(chunk[2])

vector_model=SentenceTransformer("all-MiniLM-L6-v2")
embedded=vector_model.encode(chunk)

#rint(embeddings[:1])

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="Machinelearningtable")
collections = client.list_collections()


collection.add(
    documents=chunk,
    embeddings=embedded.tolist(),
    ids=[f"id_{i}" for i in range(len(chunk))],
    metadatas=[
        {
            "chunk_id": i,
            "filename":"simplified_machinelearning"
        }
        for i in range(len(chunk))
    
    ]
)

all_data = collection.get(include=['embeddings', 'metadatas', 'documents'])
# for i in range(5):
#    print("Document:", all_data['documents'][i])
#    print("Metadata:", all_data['metadatas'][i])
#    print("Embedding length:", len(all_data['embeddings'][i]))
#    print("-"*50)

def retrive_relavant_chunk(question):
    model=SentenceTransformer("all-MiniLM-L6-v2")
    query_vector=model.encode([question][0]).tolist()

    result=collection.query(
        query_embeddings=query_vector,
        n_results=4
    )

    return result["documents"]

def llm(final_output, question):
    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the information provided in the context below.
Write the answer in exactly 4 lines.
Each line must be a complete sentence.

If the answer is not present in the context, respond exactly with:
"I don’t know based on the provided documents."


Context:
{final_output}

Question:
{question}

Answer:
"""
    
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


question=input("please ask question about machine learning:")
final_output=retrive_relavant_chunk(question)
answer=llm(final_output,question)
print(answer)