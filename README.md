This repository contains the code which was used to create an AI-powered automatic completion tool for Solidity code. 
With the code from "Preprocessing_Solidity_Dataset.ipynb" a Solidity dataset was created. This dataset was used to
fine-tune a Large Language Model with the code from "LLM-Fine-tuning.ipynb".

The file "Preprocessing_Solidity_Dataset.ipynb" contains a comprehensive preprocessing pipeline consisting of seven steps:

1. Step "Cleaning" - Removes unnecessary parts such as comments or blank lines from each Solidity file.
2. Step "Formatting" - Converts the code in each file so that the final model only generates code in a correct format. 
3. Step "Slither Analysis" - Checks for vulnerabilities in each Solidity file of the dataset. The files are sorted by the vulnerabilities they contain, and vulnerability annotations are added to the line or construct in which the vulnerability was detected.
4. Step "Splitting" - Splits each Solidity file according to the definitions (i.e. contract, interface or library) it contains. The required imports are added to the splitted files accordingly.
5. Step "Similarity Check" - Checks for duplicate and very similar contracts and removes them accordingly.
6. Step "Solhint Fixes" - Fixes some of the best practice issues detected by Solhint in the Solidity files.
7. Step "Token Insertion" - Inserts special tokens into each remaining Solidity file that mark the end of a sequence, secure code or fill-in-the-middle (FIM) code.

The file "LLM-Fine-tuning.ipynb" contains the code for tokenization of a dataset, creation of a fill-in-the-middle 
data collator, implementation of Quantized Low Rank Adaptation (QLoRA), hyperparameter optimization with Ray Tune, 
final fine-tuning of a base model, and evaluation with Perplexity, Bilingual Evaluation Understudy (BLEU), Metric
for Evaluation of Translation with Explicit ORdering (METEOR).

The fine-tuned model and some preprocessed datasets can be found at https://huggingface.co/fbnhnsl
