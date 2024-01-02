import os

from flask import Flask, jsonify, request
from flask_cors import cross_origin,CORS

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
load_dotenv()

#Global Variables
OPENAI_API_KEY = "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"
rutapdf = "./RDP.pdf"
globalconocimiento = ""

# Set app
app = Flask(__name__)
CORS(app)

@app.route('/')
@cross_origin(origin='*')
def ServerStatus():
	return "Server Started"

def create_embeddings():
    try:
        pdf_reader = PdfReader(rutapdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
            )        
        chunks = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        print(knowledge_base)
        return knowledge_base
    except Exception as e:
        print(e)
        return "Ocurrió un error"


@app.route('/testpdf')
@cross_origin(origin='*')
def test_pdf():
    pdf_reader = PdfReader(rutapdf)
    return "Numero de paginas: " + str(len(pdf_reader.pages))

#Endpoitns

@app.route('/actualizarpdf')
@cross_origin(origin='*')
def actualizar_base():
    mensaje = "hola"
    try:
        #Se setea como variable global en el backend
        knoledgebase = create_embeddings()
        return jsonify({"mensaje": "Se actualizó la base de información"}) 
        #print(conocimiento)
    except Exception as e:
        print(e)
        return "error"
    return mensaje

@app.route('/preguntar', methods=['POST'])
@cross_origin(origin='*')
def responder():
    try:
        conocimiento = create_embeddings()
        user_question = request.json['pregunta']  
        #os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = conocimiento.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)
        return respuesta
    except Exception as e:
            print(e)
            return e


@app.route('/qatest', methods=['POST'])
@cross_origin(origin='*')
def pruebarespuesta():
    responsrfont = request.get_json()
    print(responsrfont)
    return jsonify({'mensaje':'Exito'})


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=8000)