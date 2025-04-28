import json
import os
import sys

import fire
import torch
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

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
    prompt_template: str = "",
    dataset_path='',
    save_path = 'output/',
    model_name = '',
    mode='', #[standard, fol, cot],
    dataset='',
    finetune='', #['lora', 'none']
    data_size='',
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
        if finetune == 'lora':
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
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        input_length = inputs.input_ids.shape[1]
        generated_tokens = generation_output.sequences[:, input_length:]
        output = tokenizer.decode(generated_tokens[0])
        return prompter.get_response(output)

    data = load_dataset("json", data_files=dataset_path)
    samples = data['train']
    output_path = os.path.join(save_path, f"{dataset}_{model_name}_{mode}_finetune-{finetune}-{data_size}.json")
    outputs = []
    for sample in tqdm(samples):
        instruction = 'Given a premise and conclusion, generate the first order logic form of the premises and conclusion.'
        input = sample['input']
        if dataset in ["proof","pronto"]:
            input = input.replace("Context:", "Premise:").replace("Question:", " Conclusion:")
        answer = evaluate(instruction, input, max_new_tokens=600, temperature=0.0, top_p=1.0,
                          top_k=0.0, num_beams=1)
        output = {'id': sample['id'],
                  'input': sample['input'],
                  'output': sample['output'],
                  'prediction': answer}
        outputs.append(output)
    with open(output_path, 'w') as file:
        json.dump(outputs, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
