import json
import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

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
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    dataset_path='',
    save_path = '',
    model_name = '',
    mode = '', #[standard, fol, cot],
    dataset = '',
    finetune = '', #['lora', 'none']
    tool = '' #[prover9, z3, pyke, none]
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="",
            # quantization_config=quantization_config
        )
        model = PeftModel.from_pretrained(
           model,
           lora_weights,
           torch_dtype=torch.float16,
           cache_dir="",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

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
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
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
                output = tokenizer.decode(generated_tokens)
                outputs.append(output.strip())
            return outputs

        generated_tokens = generation_output.sequences[:, input_length:]
        output = tokenizer.decode(generated_tokens[0])
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
                sample['input'] = sample['input'].replace('Context:', 'Premise:').replace('Question:', 'Conclusion:')
            question = sample['input'] + '\n'
            premise = question.split('Conclusion:')[0].split('Premise:')[-1].strip().split('\n')
            if dataset in ['proof']:
                premise = question.split('Conclusion:')[0].split('Premise:')[-1].strip().strip('\n').strip('.').split('. ')
                premise = [prem+'. ' for prem in premise] # add period to the sentences
            conclusion = question.split('Conclusion:')[-1].strip()
            question_list = premise + [conclusion]
            print(len(question_list))

            # Generate predicates
            answer = evaluate(instruction, question, max_new_tokens=128, temperature=0.0, top_p=1.0)
            generated_answer = question + answer.split('\n')[0].strip()+'\nPremise_FOL- '  # Predicates: ...; Stop word: Premise_FOL-;
            predicates = answer.split('\n')[0].strip()
            file.write(generated_answer)

            # Generate FOL
            instruction_prem = "Generate first-order logic (FOL) statements from given predicates and natural language sentences."
            for i,quest in enumerate(question_list):
                if len(quest.split(' ')) < 7:
                    answer = evaluate(instruction_prem, generated_answer, max_new_tokens=16, temperature=0.0,
                                      top_p=1.0)
                else:
                    answer = evaluate(instruction_prem, generated_answer, max_new_tokens=64, temperature=0.0, top_p=1.0)

                intermediate_fol=  answer.split(':::')[0].strip()
                # print(f"FOL_{i}: {intermediate_fol}")
                if i == len(question_list)-1:
                    intermediate_fol = answer.split('Conclusion_FOL-')[-1].split(':::')[0].strip()
                    generated_answer = generated_answer + 'Conclusion_FOL- ' + intermediate_fol + ' ::: ' + quest.strip()
                else:
                    generated_answer = generated_answer + intermediate_fol + ' ::: ' + quest.strip() + '; '

            final_answer = generated_answer.strip()
            print(f"Final Generation: {final_answer}")
            file.write(f"Final Generation: {final_answer}")
            file.write("*"*64)
            output = {'id': sample['id'],
                      'input': sample['input'],
                      'output': sample['output'],
                      'prediction': final_answer}  # to be changed based on the data
            outputs.append(output)
    with open(output_path, 'w') as file:
        json.dump(outputs, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
