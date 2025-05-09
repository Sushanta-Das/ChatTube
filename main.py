
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os
import getpass
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from flask import Flask, request, jsonify
app = Flask(__name__)
load_dotenv()
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
    )
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://sushantaopenai.openai.azure.com/"

from langchain_openai import AzureOpenAIEmbeddings

embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_EMBEDDING_DEPLOYMENT_VERSION"],
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # or your deployment
    api_version="2024-12-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

def create_vectorstore(text):
    docs = [Document(page_content=text)]
    return FAISS.from_documents(docs, embedding)

# def build_qa_chain(vectorstore):
#     memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
#     return ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )

# if __name__ == "__main__":
#     # video_id = "your_youtube_video_id_here"
#     transcript = ""
#     vs = create_vectorstore(transcript)
#     qa = build_qa_chain(vs)

#     print("\nAsk questions (type 'exit' to quit):")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = qa.invoke(user_input)
#         print("Bot:", response)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vectorstore = create_vectorstore("")
retriever = vectorstore.as_retriever()

template = """
You are an intelligent assistant. Use the following chat history and documents to answer the question.

Chat history:
{chat_history}

Documents:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = LLMChain(llm=llm, prompt=prompt)

def ask(question):
    history = memory.load_memory_variables({})["chat_history"]
    docs = retriever.get_relevant_documents(question)

    if not docs:
        context = "No relevant documents found."
    else:
        context = "\n\n".join([d.page_content for d in docs])

    inputs = {
        "chat_history": "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in history]),
        "context": context,
        "question": question
    }

    response = chain.invoke(inputs)
    memory.save_context({"input": question}, {"output": response["text"]})
    return response["text"]

# === Example Usage ===
# while True:
#     q = input("You: ")
#     if q == "exit":
#         break
#     print("Bot:", ask(q))

@app.route('/ask', methods=['POST'])
def ask_endpoint():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = ask(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)    