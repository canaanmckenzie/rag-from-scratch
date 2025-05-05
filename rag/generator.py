from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#open source decoder model f_gen()
#maps context + query to natural language response
#small instruction tuned causal LLM
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

if tokenizer.pad_token is None: #decoder models without pad_token by default
    tokenizer.pad_token = tokenizer.eos_token

#move to gpu if available
if torch.cuda.is_available():
    model = model.to("cuda")

#format prompt from query + context
def format_prompt(query, docs):
    """
    Format the prompt by combining the user's question with the retrieved documents.
    Each document is a strim from the retriever
    """
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""<s>[INST] Use the following context to answer the questions below.\n
{context}

Question: {query}\nAnswer: [/INST]"""
    return prompt

def generate_response(query, docs, max_tokens=512):
    """
    Give a query and retrieved docs, generate an answer using the LLM
    """
    prompt = format_prompt(query, docs)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024, #model specific limit
        padding="max_length"
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    #response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.split("Answer:")[-1].strip()

