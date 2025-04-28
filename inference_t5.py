# /home/rtha0021/wj84_scratch/fol-verifiers/folio_t5-large-predicate-verifier
import json
import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from transformers import (GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,)
from fol_solver.Formula import FOL_Formula

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "",
    t5_model = "",
    lora_weights: str = "",
    prompt_template: str = "text_systematic_fol",  # The prompt template to use, will default to alpaca.
    dataset_path='',
    save_path = 'output/',
    model_name = '',
    mode = '', #[standard, fol, cot],
    dataset = '',
    finetune = '', #['lora', 'none']
    tool = 'T5', #[prover9, z3, pyke, none, T5]
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)

    # load Llama model
    llama_tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        llama_model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="",
        )
        llama_model = PeftModel.from_pretrained(
           llama_model,
           lora_weights,
           torch_dtype=torch.float16,
           cache_dir="",
        )
    elif device == "mps":
        llama_model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        llama_model = PeftModel.from_pretrained(
            llama_model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        llama_model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        llama_model = PeftModel.from_pretrained(
            llama_model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        llama_model.half()  # seems to fix bugs for some users.

    llama_model.eval()

    # Load T5 model
    t5_config = AutoConfig.from_pretrained(t5_model,
        cache_dir="/home/rtha0021/wj84_scratch/ramya/.cache/",
        revision="main",
        use_auth_token=None,
    )

    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model,
                                                 cache_dir="/home/rtha0021/wj84_scratch/ramya/.cache/",
                                                 use_fast=True,
                                                 revision="main",
                                                 use_auth_token=None,)

    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        t5_model,
        from_tf=bool(".ckpt" in t5_model),
        config=t5_config,
        cache_dir="/home/rtha0021/wj84_scratch/ramya/.cache/",
        revision="main",
        use_auth_token=None,
    )

    num_added_toks = t5_tokenizer.add_tokens(['∧', '→', '∨', '¬', '∀', '∃', '⊕'])
    print('We have added', num_added_toks, 'tokens')
    t5_model.resize_token_embeddings(len(t5_tokenizer))

    embedding_size = t5_model.get_input_embeddings().weight.shape[0]

    if len(t5_tokenizer) > embedding_size:
        t5_model.resize_token_embeddings(len(t5_tokenizer))

    if t5_model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    t5_model.eval()

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        # num_beams=4,
        max_new_tokens=128,
        do_sample=False,
        num_return_sequences=1,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        #print(prompt)
        inputs = llama_tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = llama_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )

        input_length = inputs.input_ids.shape[1]
        if num_return_sequences > 1:
            outputs = []
            for i, sample_output in enumerate(generation_output.sequences):
                generated_tokens = sample_output[input_length:]
                output = llama_tokenizer.decode(generated_tokens)
                outputs.append(output.strip())
            return outputs

        generated_tokens = generation_output.sequences[:, input_length:]
        output = llama_tokenizer.decode(generated_tokens[0])
        return prompter.get_response(output)

    data = load_dataset("json", data_files=dataset_path)
    samples = data['train']
    print(len(samples))
    output_path = os.path.join(save_path, f"{dataset}_{model_name}_{mode}_finetune-{finetune}_tool-{tool}.json")
    outputs = []
    with open('./logs/temp.txt', 'a') as file:
        for sample in tqdm(samples):
            instruction = "Generate predicates for the given natural language sentences."
            if dataset in ['proof']:
                sample['input'] = sample['input'].replace('Context:', 'Premise:').replace('Question:', 'Conclusion:').replace('. ', '. \n')
            question = sample['input'].replace('Premise: \n','Premise: ').replace('\n\n','\n') + '\n'
            premise = question.split('Conclusion:')[0].split('Premise:')[-1].strip().split('\n')
            conclusion = question.split('Conclusion:')[-1].strip()
            question_list = premise + [conclusion]
            statements_count = len(sample['input'].replace('Premise: \n','Premise: ').replace('\n\n','\n').split('\n'))
            # Generate predicates
            input_pred = question + "\nPredicates: "
            answer = evaluate(instruction, input_pred, max_new_tokens=64, temperature=0.0, top_p=1.0)
            print("Generated Predicate: ", answer.split('Premise_FOL')[0].strip())

            predicates = answer.split('Premise_FOL')[0].strip()
            generated_answer = input_pred + predicates + '\nPremise_FOL- '

            # Continue FOL generation
            for i, nl in enumerate(question_list):
                instruction_fol = "Generate first-order logic (FOL) statements from given predicates and natural language sentences."
                answer = evaluate(instruction_fol, generated_answer, max_new_tokens=100, temperature=0.0, top_p=1.0) # generate fol statements
                intermediate_answer = answer.split(';')[0]  # Expected fol; stop word: ';'
                if 'Conclusion_FOL' in intermediate_answer:
                    intermediate_answer = intermediate_answer.split('Predicates')[0].split(".")[0]  # Expected stopwords: 'Predicates', '.'
                print(f"Generation {i}: {intermediate_answer}")
                file.write(f"Generation {i}: {intermediate_answer}")

                intermediate_fol = intermediate_answer.split(':::')[0].strip()
                generated_answer = generated_answer + intermediate_fol + ':::' + question_list[i] + ';'

            final_answer = generated_answer.strip()
            print(f"Final Generation: {final_answer}")
            file.write(f"Final Generation: {final_answer}")
            file.write("*"*64)
            output = {'id': sample['id'],
                      'input': sample['input'],
                      'output': sample['output'],
                      'fol': sample['fol'],
                      'prediction': final_answer}  # to be changed based on the data
            outputs.append(output)
    with open(output_path, 'w') as file:
        json.dump(outputs, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
