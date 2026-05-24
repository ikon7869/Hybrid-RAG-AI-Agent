from sentence_transformers import CrossEncoder


def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    

def re_rank_docs(model, query: str, documents):
    pairs = [[query, docs] for docs in documents]

    ranked_docs = model.predict(pairs)

    for doc, score in zip(documents, ranked_docs):
        print(f"Score: {score:.4f} | Document: {doc}")

    return [doc for _, doc in sorted(zip(ranked_docs, documents), key=lambda x: x[0], reverse=True)]
