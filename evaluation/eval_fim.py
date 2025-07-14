import os
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "0"
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pickle
from nltk.translate.meteor_score import meteor_score
from pygments.lexers import SolidityLexer
from pygments.token import Token
import evaluate
import re
import nltk
nltk.download('wordnet')

def extract_prompt_and_reference(text):
    pattern = re.compile(
        r"(.*?<\|fim_end\|>)(.*)", re.DOTALL
    )
    match = pattern.match(text)
    if not match:
        return None, None

    prompt = match.group(1)
    reference = match.group(2)
    return prompt, reference

def generate_code(model, tokenizer, prompts):
    inputs = []
    print("Tokenizing prompts...")
    for prompt in prompts:
        inputs.append(tokenizer(prompt, return_tensors="pt").to(model.device))
    print("Generating code with ", model.config._name_or_path)
    outputs = []
    for input in tqdm(inputs):
        outputs.append(model.generate(**input, max_new_tokens=256, num_beams=4, temperature=0.3, do_sample=True, pad_token_id=tokenizer.eos_token_id))

    return [tokenizer.decode(output[0][len(input[0]):], skip_special_tokens=True) for (output, input) in zip(outputs, inputs)]

def tokenize_code(code):
    lexer = SolidityLexer()
    tokens = list(lexer.get_tokens(code))
    token_strings = []
    for token_type, token_value in tokens:
        if token_type not in (Token.Text, Token.Comment):
            token_strings.append(token_value)
    return token_strings # No stemming

# Computes the METEOR score
def compute_meteor(generated_code, reference_code):
    score = 0
    for (gen_code, ref_code) in zip(generated_code, reference_code):
        gen_tokens = tokenize_code(gen_code)
        ref_tokens = tokenize_code(ref_code)
        score += meteor_score([ref_tokens], gen_tokens)

    return score / len(generated_code)

def compute_metrics(dataset):
    results = {}
    vul_pattern = r".*\/\/ .*"

    # Loads the BLEU metric
    bleu = evaluate.load("bleu")

    eos_token = "<|end▁of▁sentence|>"

    # Generates predictions
    # The prompts that should be completed
    print("Preparing the prompts...")
    prompts = []
    references = []
    for prompt in tqdm(dataset):
        prompts.append(f"{prompt[0]}<|fim_hole|>{prompt[1]}<|fim_end|>")
        references.append(prompt[2])

    pretrained_predictions = generate_code(base_model, tokenizer, prompts)
    finetuned_predictions = generate_code(fin_model, tokenizer, prompts)

    # Computes the BLEU score by comparing the predictions with the references
    bleu_score_pretrained = 0
    bleu_score_finetuned = 0
    vulnerable_hits = 0
    for (finetuned_prediction, pretrained_prediction, reference) in zip(tqdm(finetuned_predictions), pretrained_predictions, references):
        if re.search(vul_pattern, finetuned_prediction):
            vulnerable_hits +=1                            # hit if vul_pattern is found in code fragment
        bleu_score_pretrained += bleu.compute(predictions=[pretrained_prediction], references=[reference])['bleu']
        bleu_score_finetuned += bleu.compute(predictions=[finetuned_prediction], references=[reference])['bleu']

    # The average of the scores is calculated
    bleu_score_pretrained = bleu_score_pretrained / len(references)
    bleu_score_finetuned = bleu_score_finetuned / len(references)

    # Computes the METEOR score by comparing the predictions with the references
    meteor_score_pretrained = compute_meteor(pretrained_predictions, references)
    meteor_score_finetuned = compute_meteor(finetuned_predictions, references)

    results[i] = {
        "bleu_pretrained": bleu_score_pretrained,
        "bleu_finetuned": bleu_score_finetuned,
        "meteor_pretrained": meteor_score_pretrained,
        "meteor_finetuned": meteor_score_finetuned,
        "vulnerable_hits": vulnerable_hits
    }

    return results

print("Preparing fim datasets...")
dataset = json.load(open("fim_dataset.json", 'r'))
data_fim = []
for data in dataset['test']:
    if data['fim_transformed'] == 1:
        data_fim.append(data)

prompts = []
references = []
for data in data_fim:
    prompt, reference = extract_prompt_and_reference(data['code'])
    if prompt and reference:
        prompts.append(prompt.strip())
        references.append(reference.strip())

import_inputs = []
for prompt, reference in zip(prompts, references):
    if 'import' in prompt or 'import' in reference:
        prefix_index = prompt.find('<|fim_hole|>')
        suffix_index = prompt.find('<|fim_end|>')
        prefix = prompt[0:prefix_index]
        suffix = prompt[prefix_index + len('<|fim_hole|>'):suffix_index]
        import_inputs.append((prefix, suffix, reference))

modifier_inputs = []
for prompt, reference in zip(prompts, references):
    if 'modifier' in prompt or 'modifier' in reference:
        prefix_index = prompt.find('<|fim_hole|>')
        suffix_index = prompt.find('<|fim_end|>')
        prefix = prompt[0:prefix_index]
        suffix = prompt[prefix_index + len('<|fim_hole|>'):suffix_index]
        modifier_inputs.append((prefix, suffix, reference))

secure_function_inputs = []
for prompt, reference in zip(prompts, references):
    if '<|secure_function|>' in prompt or '<|secure_function|>' in reference:
        prefix_index = prompt.find('<|fim_hole|>')
        suffix_index = prompt.find('<|fim_end|>')
        prefix = prompt[0:prefix_index]
        suffix = prompt[prefix_index + len('<|fim_hole|>'):suffix_index]
        secure_function_inputs.append((prefix, suffix, reference))

vulnerable_function_inputs = []
for prompt, reference in zip(prompts, references):
    if 'function' in prompt or 'function' in reference and prompt not in [input[0] for input in secure_function_inputs]:
        prefix_index = prompt.find('<|fim_hole|>')
        suffix_index = prompt.find('<|fim_end|>')
        prefix = prompt[0:prefix_index]
        suffix = prompt[prefix_index + len('<|fim_hole|>'):suffix_index]
        vulnerable_function_inputs.append((prefix, suffix, reference))

print("Initializing the models...")
base_model_id = "deepseek-ai/deepseek-coder-1.3b-base"
adapter_path = "./Finetuned_deepseek-coder-1.3b-base_Solidity_Constructs"  # directory with adapter_model.safetensors

# Load base tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model_to_be_adapted = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer.model_max_length = base_model.config.max_position_embeddings
# Load adapter on top of base model
base_model.resize_token_embeddings(len(tokenizer))
model_to_be_adapted.resize_token_embeddings(len(tokenizer))
fin_model = PeftModel.from_pretrained(model_to_be_adapted, adapter_path)

base_model.eval()
fin_model.eval()    

print("Evaluating the finetuned model on the FIM dataset...")


test_datasets = {
    "secure_function": secure_function_inputs,
    "vulnerable_function": vulnerable_function_inputs,
    "import": import_inputs,
    "modifier": modifier_inputs
}
num_rounds = 10
final_results = {}
for name, dataset in tqdm(test_datasets.items()):
    print(f"\nEvaluating dataset with {name} samples...")
    results = {}
    for i in tqdm(range(num_rounds)):
        print(f"\nRound {i + 1}/{num_rounds}")
        results[i] = compute_metrics(dataset)

    final_results[name] = results

# Save the results to a file
with open("fim_results.json", "w") as f:
    json.dump(final_results, f, indent=4)
print("Results saved to fim_results.json")