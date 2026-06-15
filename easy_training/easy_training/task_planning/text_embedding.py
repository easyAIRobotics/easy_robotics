from sentence_transformers import SentenceTransformer

class ClassTextEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)

if __name__ == "__main__":
    embedding_model = ClassTextEmbedding()
    class_names = ["red block", "blue block", "green block", "yellow block"]
    embeddings = embedding_model.encode(class_names)
    print(embeddings)