import os
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "0"
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

# Generates predictions
def generate_code(model, tokenizer, prompts):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    gens = []
    for prompt in tqdm(prompts):
        gens.append(generator(prompt, max_new_tokens=256, do_sample=True, num_beams=4, temperature=0.3, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"])
    return gens


def compute_metrics(dataset, num_rounds):
    results = {}
    vul_pattern = r".*\/\/ .*"

    # Loads the BLEU metric
    bleu = evaluate.load("bleu")

    # The prompts that should be completed
    prompts = [data[0] for data in dataset]
    references = [data[1] for data in dataset]

    for i in range(num_rounds):
        print(f"Round {i + 1}/{num_rounds}")
        # Generate predictions using the base model and the finetuned model
        pretrained_predictions = generate_code(base_model, tokenizer, prompts)
        finetuned_predictions = generate_code(fin_model, tokenizer, prompts)

        bleu_score_pretrained = 0
        bleu_score_finetuned = 0
        vulnerable_hits = 0
        for (finetuned_prediction, pretrained_prediction, reference) in zip(tqdm(finetuned_predictions), pretrained_predictions, references):
            if re.search(vul_pattern, finetuned_prediction):
                vulnerable_hits += 1
            if reference.strip() == "":
                continue
            bleu_score_pretrained += bleu.compute(predictions=[pretrained_prediction], references=[reference])['bleu']
            bleu_score_finetuned += bleu.compute(predictions=[finetuned_prediction], references=[reference])['bleu']

        bleu_score_pretrained = bleu_score_pretrained / len(references)
        bleu_score_finetuned = bleu_score_finetuned / len(references)

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

print("Preparing datasets...")
dataset = json.load(open("fim_dataset.json", 'r'))
data_non_fim = []
for data in dataset['test']:
    if data['fim_transformed'] == 0:
        data_non_fim.append(data)

split_dataset_non_fim = []
for data in data_non_fim:
    code = data['code']
    if "function" in code or "constructor" in code:
        index = code.find("{")
        if index != -1:
            prompt = code[:index + 1]
            ref = code[index + 1:]
            split_dataset_non_fim.append((prompt, ref))
        else:
            prompt = code[:code.find("(") + 1]
            ref = code[code.find("(") + 1:]
            split_dataset_non_fim.append((prompt, ref))
    elif "event" in code:
        index = code.find("{")
        if index != -1:
            prompt = code[:index + 1]
            ref = code[index + 1:]
            split_dataset_non_fim.append((prompt, ref))
        else:
            prompt = code[:code.find("(") + 1]
            ref = code[code.find("(") + 1:]
            split_dataset_non_fim.append((prompt, ref))
    else:
        if len(code) > 40:
            prompt = code[:40]
            ref = code[40:]
            split_dataset_non_fim.append((prompt, ref))
        else:
            prompt = code[:25]
            ref = code[25:]
            split_dataset_non_fim.append((prompt, ref))


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

print("Evaluating the models...")

final_results = compute_metrics(split_dataset_non_fim, 10)

# Save the results to a file
with open("non_fim_results.json", "w") as f:
    json.dump(final_results, f)
print("Results saved to non_fim_results.json")