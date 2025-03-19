import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import os
from peft import LoraConfig, get_peft_model
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Any, Literal
from IPython.display import Image
from langchain_core.runnables.graph import MermaidDrawMethod

# Load model and tokenizer
base_checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu" # cpu or gpu depend on availability

base_tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
base_model = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device) 

login("xxx")
# Load model and tokenizer
finetuned_checkpoint = "pearl41/Pearl_finetuned_smolLM"

finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_checkpoint)
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_checkpoint).to(device)

# Set your API keys
os.environ['OPENAI_API_KEY'] = 'xxx'
os.environ['PINECONE_API_KEY'] = 'xxx'

# Setup Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Load the actual Pinecone index
index_name = "review-embedding-100"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=768)

# Initialize the vector store
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def gen_response(prompt, model, tokenizer, max_new_tokens=512, temperature=0.7):
    # Tokenize the prompt, with padding and truncation enabled
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Extract input_ids and generate attention_mask explicitly
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.get("attention_mask", None).to(model.device)  # Ensure attention mask is passed

    # Generate the output using the model
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        attention_mask=attention_mask,  # Pass the attention mask explicitly
        do_sample=True,  # Enable sampling since we're using temperature
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated output and strip the prompt portion
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()  # Strip the echoed prompt part

# State Schema
class GraphState(TypedDict):
    query: str
    retriever: Any
    retrieved_reviews: List[str]
    most_relevant_review: str
    review_summary: str
    rating_prediction: str
    use_summarizer: bool
    model_type: Literal["base", "finetuned"]
    use_rating_only: bool  # Add flag to control the flow


def get_model_tokenizer(model_type):
    '''Select Model'''
    if model_type == "finetuned":
        model = finetuned_model
        tokenizer = finetuned_tokenizer
    else:
        model = base_model
        tokenizer = base_tokenizer
    return model, tokenizer

def retriever_agent(state):
    '''retrieve vectordb'''
    query = state["query"]
    retrieved_docs = retriever.get_relevant_documents(query)
    state["retrieved_reviews"] = retrieved_docs
    return state

# Agent 2: Filter relevant review
def filter_agent(state):
    """Agent filters most relevant review"""
    reviews = state["retrieved_reviews"]
    query = state["query"]
    
    # Convert Document list to plain text content
    reviews_text = "\n".join([f"Review {i+1}: {doc.page_content}" for i, doc in enumerate(reviews)])

    prompt_template = PromptTemplate.from_template(
        '''Given the query:\n"{query}"\n\nSelect the most relevant review from the list below:\n\n{reviews}\n\nRespond only with the most relevant review.'''
    )
    final_prompt = prompt_template.format(query=query, reviews=reviews_text)
    
    response = llm([HumanMessage(content=final_prompt)])
    state["most_relevant_review"] = response.content  # .content is required here
    return state

def summarizer_agent(state):
    '''agent summerize most relevant review'''
    review = state["most_relevant_review"]

    prompt_template = PromptTemplate.from_template(
        '''Summarize the following review clearly and concisely:\n\n"{review}"\n\nSummary:'''
    )
    prompt = prompt_template.format(review=review)

    # Generate using OpenAI LLM
    response = llm.predict(prompt)
    state["review_summary"] = response.strip()
    return state

def rating_agent(state):
    '''agent predict rating'''
    # Get model type and use_summarizer flag from the state
    model_type = state.get("model_type", "base")
    use_summarizer = state.get("use_summarizer", False)

    # Select review text based on use_summarizer flag
    review = state.get("review_summary", "") if use_summarizer else state.get("most_relevant_review", state.get("query", ""))

    # Load model + tokenizer
    model, tokenizer = get_model_tokenizer(model_type)

    # Build prompt
    prompt = f'''
Given the review below, estimate a likely rating (1 to 5):
Respond in the format:\nEstimated Rating: <1-5>

"{review}"
'''

    # Generate response using the model
    try:
        response = gen_response(prompt, model, tokenizer).strip()
        state["rating_prediction"] = response
    except Exception as e:
        state["rating_prediction"] = "I'm not trained for this..."

    return state

def should_use_summarizer(state):
    return "SummarizerAgent" if state.get("use_summarizer", False) else "RatingAgent"

def should_skip_to_rating_agent(state):
    return "RatingAgent" if state.get("use_rating_only", False) else "RetrieverAgent"

# Define workflow using LangGraph
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("RetrieverAgent", retriever_agent)
workflow.add_node("FilterAgent", filter_agent)
workflow.add_node("SummarizerAgent", summarizer_agent)
workflow.add_node("RatingAgent", rating_agent)

# Conditional logic to decide if we skip everything and go directly to RatingAgent
workflow.add_conditional_edges(START, should_skip_to_rating_agent, {
    "RatingAgent": "RatingAgent",   # Directly go to RatingAgent
    "RetrieverAgent": "RetrieverAgent"  # Proceed with the usual path
})

# Proceed with FilterAgent always after RetrieverAgent
workflow.add_edge("RetrieverAgent", "FilterAgent")

# Conditional logic to determine if summarization should occur or not after FilterAgent
workflow.add_conditional_edges("FilterAgent", should_use_summarizer, {
    "SummarizerAgent": "SummarizerAgent",   # Proceed to SummarizerAgent if summarizer is enabled
    "RatingAgent": "RatingAgent"            # Skip summarizer and go to RatingAgent if summarizer is disabled
})

# Continue to RatingAgent after SummarizerAgent if summarization is used
workflow.add_edge("SummarizerAgent", "RatingAgent")

# Finally, go to the end after RatingAgent
workflow.add_edge("RatingAgent", END)

# Compile the graph
app = workflow.compile()