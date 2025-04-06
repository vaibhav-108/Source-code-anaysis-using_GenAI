# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
import shutil
import stat
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
# from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)


load_dotenv()

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

HUGGINGFACE_API_KEY= os.environ.get("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"]=HUGGINGFACE_API_KEY


embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# llm = ChatOpenAI()
llm= llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.5,
    task='text-generation',
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        print(user_input)
        repo_ingestion(user_input)
        # os.system("python store_index.py")  #--> to execute stor_index.py

    return jsonify({"response": str(user_input) })



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rmdir /s /q repo")
        # shutil.rmtree("repo", ignore_errors=True)
        # os.system("rm -rf test_repo")
        #-- To delete file---#
        # def remove_readonly(func, path, _):
        #     """Clear the readonly bit and reattempt the removal."""
        #     os.chmod(path, stat.S_IWRITE)
        #     func(path)


        # shutil.rmtree("test_repo", onexc=remove_readonly)

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)