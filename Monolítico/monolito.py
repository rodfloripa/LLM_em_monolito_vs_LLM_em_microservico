import flask
import json
import torch
import faiss
import numpy as np
import sys
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = flask.Flask(__name__)

# --- CONFIGURAÇÃO DE CAMINHOS ---
# O Dockerfile copiará a pasta model_weights para /app/model_weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights")

print(f"DEBUG - Carregando modelo local de: {MODEL_PATH}", file=sys.stderr)

try:
    # local_files_only=True garante que o código não tente conectar ao HuggingFace
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print("DEBUG - Modelo carregado com sucesso!", file=sys.stderr)
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar os pesos locais: {str(e)}", file=sys.stderr)
    sys.exit(1)

# --- BASE DE CONHECIMENTO (RAG) ---
docs = [
    "O clima no Brasil é predominantemente tropical.",
    "São Paulo é o maior centro financeiro da América Latina.",
    "O SageMaker Serverless escala automaticamente conforme a demanda.",
    "A Floresta Amazônica é vital para o equilíbrio climático global."
]

# Inicializa o FAISS com dimensão 512 (padrão do T5-small)
index = faiss.IndexFlatL2(512)

def get_embedding(text):
    """Gera representação vetorial usando o encoder do modelo."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=inputs.input_ids)
        # Média das saídas para gerar o vetor fixo
        vector = encoder_outputs.last_hidden_state.mean(dim=1).numpy()
    return vector

print("DEBUG - Indexando documentos no FAISS...", file=sys.stderr)
for d in docs:
    index.add(get_embedding(d))

def perform_rag(query):
    """Processo completo de Retrieval e Generation."""
    # 1. Retrieval (Busca)
    query_vector = get_embedding(query)
    _, I = index.search(query_vector, k=1)
    context = docs[I[0][0]]
    
    # 2. Generation (Resposta)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- ROTAS SAGEMAKER ---

@app.route('/ping', methods=['GET'])
def ping():
    """Obrigatório para o SageMaker saber que o container está saudável."""
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Entrada principal para inferência."""
    data = flask.request.get_json(force=True, silent=True)
    if not data or 'query' not in data:
        return flask.jsonify({"error": "Campo 'query' não encontrado"}), 400

    query = data['query']
    
    try:
        resposta = perform_rag(query)
        result = {
            "pergunta": query, 
            "resposta": resposta
        }
        # ensure_ascii=False para exibir caracteres PT-BR corretamente
        return flask.Response(
            response=json.dumps(result, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        print(f"ERRO durante inferência: {str(e)}", file=sys.stderr)
        return flask.jsonify({'error': 'Erro ao processar RAG'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
