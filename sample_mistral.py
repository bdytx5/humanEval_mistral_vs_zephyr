import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems

# Set the device to GPU (CUDA) if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Mistral-7B-Instruct-v0.1 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model.eval()
i = 0 

def split_and_trim_code(text: str):
    # Find the index where "def " starts
    def_index = text.find('def ')
    if def_index == -1:
        return ""  # return an empty string if "def " is not found

    # Extract the code starting from "def "
    code_after_def = text[def_index:]

    # Find the end of the "def " line
    def_line_end_index = code_after_def.find('\n')
    if def_line_end_index != -1:
        # Skip the entire line where "def " is found
        code_after_def = code_after_def[def_line_end_index+1:]
    
    # Find the index of "```" that comes after "def "
    end_code_index = code_after_def.find('```')
    
    # If "```" is found, return the code until just before "```"
    if end_code_index != -1:
        code = code_after_def[:end_code_index]
    else:
        # If no "```" is found, return the code until the end of the string
        code = code_after_def
    
    # Removing leading and trailing whitespaces and newlines
    return code.strip()



# Define a function to generate one completion
def generate_one_completion(prompt):
    global i 
    i += 1
    print("Generating")
    print(i)

    # Wrapping the prompt with instruction format tokens
    formatted_prompt = f"<s>[INST] You are a friendly chatbot. WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!! {prompt} [/INST]"

    # Generate a response using the model
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = None 
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated ids to text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the text split and trimmed
    print("###" * 50)
    print(split_and_trim_code(text.split("[/INST]")[1]))
    return split_and_trim_code(text.split("[/INST]")[1])

# Read problems from the dataset
problems = read_problems()

# Define the number of samples to generate for each problem
num_samples_per_task = 10

# Generate samples
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problem["prompt"]))
    for task_id, problem in problems.items()
    for _ in range(num_samples_per_task)
]

# Save generated samples in jsonl format
write_jsonl("mistral_samples.jsonl", samples)

