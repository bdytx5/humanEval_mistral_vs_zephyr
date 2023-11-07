import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems

# Set the device to GPU (CUDA) if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# torch.set_default_device(device)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

def generate_one_completion(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as model

    # Generate completion using the model
    outputs = model.generate(**inputs, max_length=200)
    
    # Decode the generated tokens to text
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return text

# Read problems from the dataset
problems = read_problems()

# Define number of samples to generate for each problem
num_samples_per_task = 1

# Generate samples
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]

# Save generated samples in jsonl format
write_jsonl("samples.jsonl", samples)

# To evaluate the samples, you would run the evaluate_functional_correctness command in your terminal as described in the README.
