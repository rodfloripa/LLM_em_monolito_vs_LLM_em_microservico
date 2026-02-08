# Usamos a versão slim para manter o container leve e rápido no Cold Start
FROM python:3.9-slim

# Instala dependências do sistema necessárias para o FAISS e processamento numérico
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala as dependências do Python primeiro (aproveita o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia a pasta de pesos do modelo (que você criou com o comando anterior)
# Isso garante que o modelo esteja "embutido" e não dependa de internet no boot
COPY model_weights/ /app/model_weights/

# Copia o código principal
COPY monolito.py .

# Configurações de ambiente para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Porta padrão que o SageMaker escuta
EXPOSE 8080

# Usamos o Gunicorn como servidor de produção. 
# Aumentamos o timeout para 120s para dar tempo do modelo carregar na RAM sem dar erro.
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "120", "monolito:app"]
