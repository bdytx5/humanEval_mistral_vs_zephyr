import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems

# Set the device to GPU (CUDA) if available
from transformers import pipeline

# Initialize the pipeline for text generation with the Zephyr model
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

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



i = 0 
def generate_one_completion(prompt):
    # Format the prompt using the tokenizer's chat template
    global i 
    i+=1 
    print("Generating {}".format(i))
    messages = [
        {
            "role": "system",
            # "content": "You are a friendly chatbot. Complete the remaining portion of the code. ONLY WRITE THE REMAINING PORTION. DO NOT EXPLAIN THE CODE. DO NOT WRITE THE FULL FUNCTION, ONLY THE REMAINING PORTION. END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",
            "content": "You are a friendly chatbot. WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",

        },
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate a response using the model
    outputs = pipe(formatted_prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95)
    
    # Extract the generated text
    text = outputs[0]["generated_text"]

    print("###"*50)
    # print(split_and_trim_code(text.split("<|assistant|>")[1]))
    # return text
    return split_and_trim_code(text.split("<|assistant|>")[1])


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
write_jsonl("zephyr_samples.jsonl", samples)

