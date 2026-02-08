# LLM_em_monolito_vs_LLM_em_microservico

# Comparativo de Arquiteturas: Monolítica vs. Microserviços (RAG)
<p align="center">
  <img src="https://github.com/rodfloripa/LLM_em_monolito_vs_LLM_em_microservico/blob/main/Monolitico.png">
</p>
<p align="center">Fig1. Arquitetura Monolítica</p>
<br><br>
<p align="center">
  <img src="https://github.com/rodfloripa/LLM_em_monolito_vs_LLM_em_microservico/blob/main/Microsservi%C3%A7os.png">
</p>
<p align="center">Fig2. Arquitetura com Microserviços</p>
<br><br>



<div align="justify">

A transição entre uma arquitetura monolítica e uma baseada em microserviços é um marco na jornada de qualquer engenheiro de software. No contexto de um sistema **RAG (Retrieval-Augmented Generation)** rodando no AWS SageMaker, essa escolha impacta diretamente a estabilidade, o custo e a eficiência do hardware. Abaixo, detalhamos as vantagens e desvantagens de cada abordagem.

<p align="left">
Como rodar:
  
1. Faz o build, envia para o ECR e atualiza o endpoint no SageMaker:

  make build && make push && make deploy

2. Aguarde 15 minutos e digite make status,aguardando até ficar InService

3. make test 
</p>
---

## 1. Arquitetura Monolítica
*Nesta estrutura, o script de inferência, o modelo de busca vetorial e o LLM compartilham o mesmo processo e a mesma máquina.*

| **Vantagens** | **Desvantagens** |
| :--- | :--- |
| **Simplicidade de Fluxo:** Mais fácil de desenvolver e testar localmente, exigindo apenas um único arquivo de inicialização. | **Risco de Crash em Cascata:** Se o LLM consumir muita RAM e causar um erro de Out of Memory (OOM), todo o sistema cai. |
| **Baixa Latência Interna:** Não existe o overhead de chamadas HTTP entre a busca e a geração, pois os dados estão na mesma memória. | **Desperdício de Hardware:** Você é obrigado a rodar tudo em uma instância cara com GPU, mesmo para tarefas que só exigem CPU. |
| **Boot Mais Rápido:** Apenas um servidor Gunicorn precisa ser iniciado, reduzindo o tempo inicial de carregamento do container. | **Acoplamento Forte:** Um erro de biblioteca no serviço de texto impede o funcionamento da busca vetorial. |



---

## 2. Arquitetura de Microserviços
*O Orquestrador gerencia o Vector Service (5001) e o LLM Service (5002) como entidades independentes.*

| **Vantagens** | **Desvantagens** |
| :--- | :--- |
| **Otimização de Hardware (Heterogêneo):** Permite rodar o LLM em instâncias com **GPU** (acelerando a geração) enquanto mantém o Vector Service e o Orquestrador em **CPU** (reduzindo custos). | **Complexidade Operacional:** Exige gerenciamento de múltiplas portas, scripts de inicialização (`start.sh`) e orquestração de saúde. |
| **Isolamento de Falhas:** O Vector Service pode estar ativo e pronto enquanto o LLM ainda está carregando ou sendo corrigido. | **Overhead de Rede:** Cada troca de informação entre os serviços adiciona alguns milissegundos devido ao protocolo HTTP interno. |
| **Escalabilidade Independente:** É possível escalar apenas o serviço de LLM se houver muitas requisições, sem precisar duplicar a base vetorial. | **Maior Consumo de RAM Base:** Manter três servidores Flask/Gunicorn ativos simultaneamente exige mais memória base. |



---

## Conclusão de Engenharia

Para o projeto **RAG Microservices**, a arquitetura de microserviços provou-se superior não apenas pela resiliência, mas pela flexibilidade estratégica. Em um cenário de produção, poderíamos direcionar o **LLM Service** para um cluster de GPUs (como instâncias `ml.p3`), enquanto o **Vector Service** e o **Orquestrador** rodariam de forma econômica em CPUs (instâncias `ml.t3`), otimizando o custo por requisição sem sacrificar a velocidade de resposta do modelo.

</div>
