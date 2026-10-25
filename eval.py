import sys
import langchain_google_vertexai
sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision


from hybrid_rerank_rag import hybrid_search, rerank, groq_client


MODEL = "qwen/qwen3.8-27b"
eval_llm = ChatGroq(model=MODEL, temperature=0, api_key=groq_client.api_key)


metrics = [Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision()]
judge_llm = LangchainLLMWrapper(eval_llm)
judge_emb = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)


test_questions = [
    "What is the annual salary range for an AI Engineer in 2026?",
    "What is the annual salary range for a Project Manager in 2026?",
    "How much net profit was reached in the first quarter?",
    "What percentage of total profits will be allocated to AI infrastructure and cloud expansion?",
    "How much profit will be allocated to AI infrastructure?",
    "How much is the outstanding partner debt?",
    "When is the partner debt due?",
    "How are annual performance bonuses determined?",
    "When is the Q4 evaluation for performance bonuses?",
    "What is the maximum salary for an AI Engineer?",
    "What is the minimum salary for a Project Manager?",
    "Is the partner debt larger or smaller than the AI infrastructure allocation?",
]
ground_truths = [
    "An AI Engineer earns $80,000 to $120,000 per year in 2026.",
    "A Project Manager earns $90,000 to $130,000 per year in 2026.",
    "Net profit reached $1.2 million in the first quarter.",
    "30% of total profits will be allocated to AI infrastructure and cloud expansion.",
    "$360,000 (30% of $1.2 million net profit) will be allocated to AI infrastructure.",
    "The outstanding partner debt is $200,000.",
    "The partner debt is due by the end of the fiscal year.",
    "Annual performance bonuses are determined based on the Q4 evaluation in December.",
    "The Q4 evaluation takes place in December.",
    "The maximum salary for an AI Engineer is $120,000 per year.",
    "The minimum salary for a Project Manager is $90,000 per year.",
    "The partner debt ($200,000) is smaller than the AI infrastructure allocation ($360,000).",
]

answers, contexts = [], []

print(" Running RAG Pipeline to collect evaluation data...")
for query in test_questions:
    candidates = hybrid_search(query, top_k=3)
    top_contexts = rerank(query, candidates, top_n=2)
    contexts.append(top_contexts)

    context_text = "\n".join(top_contexts)
    prompt = f"""You are a precise AI assistant. Answer the user's question clearly and directly using ONLY the context provided below.
Do NOT explain the math or steps—state the final requested fact/number in a complete sentence.


Context:
{context_text}

Question: {query}
Answer:"""
    response = eval_llm.invoke(prompt)
    answers.append(response.content)


dataset = Dataset.from_dict({
    "user_input": test_questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths,
})

print(" Calculating RAGAS Evaluation Scores...")
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=judge_emb,
    run_config=RunConfig(timeout=180, max_workers=2),
)

print("\n===  RAGAS EVALUATION METRICS RESULTS ===")
print(results)
print("\n", results.to_pandas())