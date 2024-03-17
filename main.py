from langchain_community.llms import LlamaCpp, HuggingFaceHub
from langchain_community.vectorstores import Qdrant
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

srcDir = "/Users/jeganprakash/Files/ds/HashMap/src/main/kotlin"
transformedDir = "/Users/jeganprakash/Creative/transformedHashmap"

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def convert_files_to_txt(src_dir, dst_dir):
    # If the destination directory does not exist, create it.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ['build', '.git', 'classes']]
        for file in files:
            if not file.endswith(('.jpg', '.class', '.jar')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                # get the relative path to preserve directory structure
                # Create the same directory structure in the new directory
                new_root = os.path.join(dst_dir, os.path.dirname(rel_path))
                os.makedirs(new_root, exist_ok=True)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                    except UnicodeDecodeError:
                        print(f"Failed to decode the file: {file_path}")
                        continue
                # Create a new file path with .txt extension
                new_file_path = os.path.join(new_root, file + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(data)



def pretty_print_docs(documents):
    for doc in documents:
        print(doc.metadata)
        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        print(doc.page_content)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    src_dir = transformedDir

   # convert_files_to_txt(srcDir, transformedDir)

    loader = DirectoryLoader(src_dir, show_progress=True, loader_cls=TextLoader)
    repo_files = loader.load()
    print(f"Number of files loaded: {len(repo_files)}")
    #
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    documents = text_splitter.split_documents(documents=repo_files)
    print(f"Number of documents : {len(documents)}")

    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                          model_kwargs=model_kwargs,
                                          encode_kwargs=encode_kwargs,
                                          )

    gpu_llm = HuggingFacePipeline.from_model_id(
        model_id="codellama/CodeLlama-7b-hf",
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 100},
    )

    print_hi('Supper')

    qdrant = Qdrant.from_documents(
        documents,
        embeddings,
        path="/Users/jeganprakash/Creative/hasMap_local_qdrant",
        collection_name="my_documents",
    )

    print_hi('PyCharm')

    query = "what is the syntax to add a value in hashMap"
    found_docs = qdrant.similarity_search(query)
   # pretty_print_docs(found_docs)

    from langchain import hub

    from langchain import hub

    prompt = hub.pull("rlm/rag-prompt")
    Prompt: ChatPromptTemplate(input_variables=['context', 'question'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
    chain = prompt | gpu_llm | StrOutputParser()
    response = chain.invoke({"question": query, "context": found_docs})
    print(response)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
