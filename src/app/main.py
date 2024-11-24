from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient
from vanna.flask import VannaFlaskApp

class MyVanna(Qdrant_VectorStore, Ollama):
    def __init__(self, config=None):
        
        if config is None:
            config = {}
            
            
        qdrant_client = QdrantClient(url="localhost", port=6333)
        
        qdrant_config = {'client': qdrant_client}
        ollama_config = {
            'model': 'llama2:7b',  # Specify the model name
            'url': 'http://localhost:11434',  # Default Ollama API endpoint
            'temperature': 0.7,  # Optional: adjust temperature for response randomness
            'num_ctx': 4096,     # Context window size
            'num_thread': 4      # Number of threads to use
        }
        
        config = {**qdrant_config, **ollama_config, **config}
        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


if __name__ == "__main__":
    vn = MyVanna()


    app = VannaFlaskApp(vn)
    
    # Configurando timeout
    app.config['TIMEOUT'] = 900
    app.run()