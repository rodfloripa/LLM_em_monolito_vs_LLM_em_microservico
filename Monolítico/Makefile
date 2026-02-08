# --- VARIÁVEIS DO PROJETO ---
ACCOUNT_ID=634510061385
REGION=us-east-1
IMAGE_NAME=rag-monolito
ENDPOINT_NAME=rag-monolito-endpoint
ROLE_ARN=arn:aws:iam::634510061385:role/RodneySageMakerRole

# --- FLUXO PRINCIPAL ---
deploy: prepare build push create-model create-config create-endpoint

# 1. PREPARAÇÃO: Baixa os pesos do modelo para uma pasta local usando Docker
# Isso evita problemas de dependências na sua máquina e resolve o erro 401 no SageMaker
prepare:
	@echo "Verificando pesos do modelo..."
	@if [ ! -d "./model_weights" ]; then \
		echo "Baixando t5-small para ./model_weights..."; \
		docker run -v $(shell pwd):/app -w /app python:3.9-slim /bin/bash -c "\
			pip install --no-cache-dir transformers torch && \
			python3 -c \"from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
			m='t5-small'; \
			AutoModelForSeq2SeqLM.from_pretrained(m, token=False).save_pretrained('./model_weights'); \
			AutoTokenizer.from_pretrained(m, token=False).save_pretrained('./model_weights')\""; \
	else \
		echo "Pesos já existem localmente. Pulando download."; \
	fi

# 2. BUILD: Constrói a imagem Docker com os pesos embutidos
build:
	docker build -t $(IMAGE_NAME) .

# 3. PUSH: Envia para o Amazon ECR
push:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com
	aws ecr create-repository --repository-name $(IMAGE_NAME) || true
	docker tag $(IMAGE_NAME):latest $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_NAME):latest
	docker push $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_NAME):latest

# 4. MODEL: Cria a entidade do modelo no SageMaker
create-model:
	-aws sagemaker delete-model --model-name $(IMAGE_NAME)
	aws sagemaker create-model --model-name $(IMAGE_NAME) \
		--primary-container Image=$(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(IMAGE_NAME):latest \
		--execution-role-arn $(ROLE_ARN)

# 5. CONFIG: Define a configuração Serverless (6GB para aguentar o RAG)
create-config:
	-aws sagemaker delete-endpoint-config --endpoint-config-name $(IMAGE_NAME)-config
	aws sagemaker create-endpoint-config --endpoint-config-name $(IMAGE_NAME)-config \
		--production-variants VariantName=AllTraffic,ModelName=$(IMAGE_NAME),ServerlessConfig={MemorySizeInMB=6144,MaxConcurrency=2}

# 6. ENDPOINT: Sobe o serviço real
create-endpoint:
	-aws sagemaker delete-endpoint --endpoint-name $(ENDPOINT_NAME)
	aws sagemaker create-endpoint --endpoint-name $(ENDPOINT_NAME) --endpoint-config-name $(IMAGE_NAME)-config

# 7. TESTE: Invocação rápida
test:
	@echo '{"query": "Qual a capital do Brasil?"}' > input.json
	aws sagemaker-runtime invoke-endpoint --endpoint-name $(ENDPOINT_NAME) --body fileb://input.json --content-type application/json saida.json
	cat saida.json
	@echo ""

# 8. DESTROY: Limpa tudo
destroy:
	-aws sagemaker delete-endpoint --endpoint-name $(ENDPOINT_NAME)
	-aws sagemaker delete-endpoint-config --endpoint-config-name $(IMAGE_NAME)-config
	-aws sagemaker delete-model --model-name $(IMAGE_NAME)
