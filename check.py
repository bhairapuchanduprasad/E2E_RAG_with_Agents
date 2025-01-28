import json
import os
from qdrant_client import QdrantClient
from openai import OpenAI

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()

# Qdrant Client Configuration
qdrant_client = QdrantClient(
    url="https://a1665047-1038-4e5d-98b8-c6ee68c82a8e.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key="9c6J_am3jiuQyzAUBF8nb0GAKn1clxmaYeChZ61ze0C6kbUA6FK4QQ",
     verify = False
)
COLLECTION_NAME = "my_collection_hr_2"
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate, RunConfig
from ragas.metrics import context_recall, context_precision, faithfulness, answer_correctness
from ragas import evaluate, RunConfig
from ragas.metrics import context_recall, context_precision, faithfulness, answer_correctness
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

def evaluate_ragas(question, answer, chunks):
    # Prepare the dataset for evaluation
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [[chunk['text'] for chunk in chunks]],
        'ground_truth': [""]  # Ground truth can be passed if available
    }

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df, features=Features({
        'question': Value('string'),
        'answer': Value('string'),
        'contexts': Sequence(Value('string')),
        'ground_truth': Value('string'),
    }))

    # Configure the OpenAI embedding model
    config = RunConfig(timeout=500)

    def embed_with_openai(texts):
        from openai.embeddings_utils import get_embedding
        return [get_embedding(text, engine="text-embedding-3-small") for text in texts]

    # Evaluate the metrics using OpenAI embeddings
    results = evaluate(dataset, metrics=[context_recall, context_precision, faithfulness, answer_correctness], llm = llm,  embeddings=embeddings, run_config=config)
    # print(results.scores) 

    return results.scores


def embed_query_with_openai(query: str):
    try:
        response = client.embeddings.create(input=query, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")

def query_qdrant(embedding, top_k: int):
    try:
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        return results
    except Exception as e:
        raise Exception(f"Error querying Qdrant: {e}")

def generate_answer_with_openai(question: str, context: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are an expert HR assistant for ChemNovus Incorporated. "
                    "Your task is to provide accurate, concise, and relevant answers strictly based on the provided context. "
                    "If the context does not contain enough information to answer the question, politely indicate that the information is not available."
                )},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=400,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error generating answer: {e}")


def classify_question(question: str) -> str:
    system_prompt = {
        "role": "system",
        "content": (
            """You are an HR assistant for ChemNovus Incorporated. Your task is to classify employee queries into the following categories:\n"
            1. Direct_Question: Questions that can be answered without internal database knowledge.\n
            2. Simple_Question: Questions that need data retrieval from ChemNovus' internal HR database but are\n
            3. Complex_Question: Questions that require breaking into sub-questions and retrieving multiple pieces of data.\n
            Provide a single classification for the query, you have to answer with either "Direct_Question", "Simple_Question", "Complex_Question" strictly.
            
            Examples 
            What are chemnovus HR policies? -> Simple_Question
            What are chemnovus HR policies and supplementary health insurance options? -> Complex_Question
            """
        )
    }
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                system_prompt,
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Error classifying question: {e}")


def decompose_complex_question(question: str) -> list:
    system_prompt = {
        "role": "system",
        "content": (
            """You are an HR assistant for ChemNovus Incorporated. Your task is to break down complex queries into simpler, self-contained sub-questions, 
                the maximum number of questions you can break it down is to 3."""
        )
    }
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                system_prompt,
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        raise Exception(f"Error decomposing question: {e}")

def generate_final_answer(question: str, responses: list) -> str:
    system_prompt = {
        "role": "system",
        "content": (
            "Combine multiple sub-answers into a coherent and relevant response to the user's original question."
        )
    }
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                system_prompt,
                {"role": "user", "content": f"Original Question: {question}\n\nSub-Answers:\n" + "\n".join(responses)}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Error generating final answer: {e}")

def local_handler(question, evaluation_flag="no", top_k=10):
    """
    Simulates the lambda_handler for local testing.
    Args:
        question (str): The question to classify and process.
        evaluation_flag (str): Set to "yes" if RAGAS evaluation is needed.
        top_k (int): Number of top results to retrieve.
    Returns:
        dict: The response containing classification, answer, and optionally evaluation.
    """
    try:
        # Classify the question
        classification = classify_question(question)

        if "direct" in classification.lower():
            answer = generate_answer_with_openai(question, "")
            response = {"classification": classification, "answer": answer, "chunks": []}

        elif "simple" in classification.lower():
            embedding = embed_query_with_openai(question)
            results = query_qdrant(embedding, top_k)
            retrieved_chunks = [
                {"text": result.payload["text"], "similarity_score": result.score}
                for result in results if "text" in result.payload
            ]
            context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
            answer = generate_answer_with_openai(question, context)
            response = {"classification": classification, "answer": answer, "chunks": retrieved_chunks}

        elif "complex" in classification.lower():
            sub_questions = decompose_complex_question(question)
            responses = []
            retrieved_chunks = []

            for sub_question in sub_questions:
                embedding = embed_query_with_openai(sub_question)
                results = query_qdrant(embedding, top_k)
                retrieved_contexts = [
                    {"text": result.payload["text"], "similarity_score": result.score}
                    for result in results if "text" in result.payload
                ]
                context = "\n\n".join(chunk["text"] for chunk in retrieved_contexts)
                responses.append(generate_answer_with_openai(sub_question, context))
                retrieved_chunks.append({
                    "sub_question": sub_question,
                    "retrieved_contexts": retrieved_contexts
                })

            final_answer = generate_final_answer(question, responses)
            response = {
                "classification": classification,
                "answer": final_answer,
                "sub_answers": responses,
                "chunks": retrieved_chunks
            }

        else:
            return {"error": "Unknown classification."}

        # Add RAGAS evaluation if requested
        if evaluation_flag.lower() == "yes":
            ragas_results = evaluate_ragas(question, response["answer"], response["chunks"])
            response["evaluation"] = ragas_results

        return response

    except Exception as e:
        return {"error": str(e)}


# Test the local handler
if __name__ == "__main__":
    question = "What are the supplementary health insurance options provided by ChemNovus Incorporated?"
    evaluation_flag = "yes"
    top_k = 3
    result = local_handler(question, evaluation_flag, top_k)
    print(result)
    print(json.dumps(result, indent=4))
