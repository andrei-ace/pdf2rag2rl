import torch
import transformers
from torch.nn.functional import cosine_similarity
from embeddings import get_text_embeddings
from graphs import extract_text_from_graph, split_graph

model_id = "meta-llama/Meta-Llama-3.1-8B"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)


def generate_answer(question, context, min_new_tokens=8, max_new_tokens=512):
    input_text = f"Using this context: {context}\n\nAnswer to the following question: {question}\n\nAnswer:"
    # Use beam search with a small number of beams to enforce brevity and relevance
    result = pipeline(
        input_text,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        min_new_tokens=min_new_tokens,
        num_beams=3,
        truncation=True,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )
    return result[0]["generated_text"].split("Answer:", 1)[1].strip()


def retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=1):
    # Compute cosine similarities
    similarities = cosine_similarity(question_embedding.unsqueeze(0), text_embeddings)
    # Sort indices of similarities in descending order
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    return " ".join([texts[idx] for idx in top_k_indices])


def evaluate_answer(generated_answer, provided_answer):
    # Get embeddings for both answers
    gen_embedding = get_text_embeddings([generated_answer])
    prov_embedding = get_text_embeddings([provided_answer])

    # Compute cosine similarity
    similarity = cosine_similarity(gen_embedding, prov_embedding)
    return similarity.item()


def rag(graph, nodes, edges, questions_answers):
    subgraphs = split_graph(graph, nodes, edges)
    texts = []
    for subgraph, nodes, edges in subgraphs:
        txt = extract_text_from_graph(subgraph, nodes, edges)
        texts.append(txt)

    text_embeddings = get_text_embeddings(texts)
    questions, answers = zip(*questions_answers)
    question_embeddings = get_text_embeddings(questions)

    results = []
    for question, provided_answer, question_embedding in zip(questions, answers, question_embeddings):
        relevant_text = retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=1)
        generated_answer = generate_answer(question, relevant_text)
        score = evaluate_answer(generated_answer, provided_answer)
        results.append((question, provided_answer, generated_answer, score))

    return results
