from langchain_redis import RedisConfig, RedisVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_redis_client(host='localhost', port=6379, db=0):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return RedisVectorStore(
        embeddings,
        config=RedisConfig(
            index_name="cached_contents",
            redis_url="redis://localhost:6379",
            distance_metric="COSINE",
            metadata_schema=[{
                "name":"llm_output", "type": "text"
            }]
        )

    )


def store_cache(user_query: str, llm_output: str):
    try:
        client = get_redis_client()

        cached_doc = Document(
            page_content=user_query,
            metadata={"llm_output": llm_output}
        )

        client.add_documents([cached_doc])

        return True
    except:
        return False


def get_cached_answer(redis_vector_store: RedisVectorStore, user_query: str, distance_thres: float = 0.15):

    results_with_scores = redis_vector_store.similarity_search_with_score(query=user_query, k=1)

    if results_with_scores:
        doc, score = results_with_scores[0]

        if score <= distance_thres:
            return doc.metadata.get("llm_output")
        
        return None
                   

