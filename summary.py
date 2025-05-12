from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import getpass
# ---------------------- Azure Config ----------------------
load_dotenv()  # Load variables from .env
if "AZURE_OPENAI_API_KEY" not in os.environ:
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOpenAI API key: "
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


template_chunks_summary = """
Return the Summary of the following video transcript in {target_lang} : {chunk} 
"""
template_final_summary ="""
            Return Summary of the following video transcript merging repeating sentences in {target_lang} : {final_summary} 
   """


prompt_chunks_summary = ChatPromptTemplate.from_template(template_chunks_summary)
prompt_final_summary = ChatPromptTemplate.from_template(template_final_summary)
chain_chunks_summary = prompt_chunks_summary | llm
chain_final_summary = prompt_final_summary | llm



# As GPT has a limit of context length of 128000 tokens, we need to split the text into smaller chunks  . each token approximately consists of 4 characters
def split_text(text, max_chars=500000):
    """Splits a long text into smaller parts by sentences."""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chars:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks


def summarize_with_translation(text, target_lang="english"):
    chunks = split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    partial_summaries = []
     
    for chunk in chunks:
       
        inputs = {
        "target_lang": target_lang,
        "chunk": chunk,
         }

        response = chain_chunks_summary.invoke(inputs)
        partial_summaries.append(response.content)
    if len(partial_summaries)>1:    
        final_summary = " ".join(partial_summaries)
        response= chain_final_summary.invoke({"target_lang": target_lang, "final_summary": final_summary})
        
   
    return  response.content