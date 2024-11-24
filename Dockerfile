FROM ollama/ollama:latest

ENV OLLAMA_SKIP_TLS_VERIFY=true

# Atualiza os repositórios e instala ca-certificates
RUN apt-get update && \
  apt-get install -y ca-certificates && \
  rm -rf /var/lib/apt/lists/*

# Atualiza os certificados
RUN update-ca-certificates