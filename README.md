# Logic_Translations
This repository presents strategies to improve natural language to first-order logic (NL-to-FOL) translations for deductive logical reasoning tasks.

## Installation

1. **Clone this repository**
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/RamyaKeerthy/Translation-NL2FOL

2. **Set Up the Environment**
  Install the required dependencies:
   ```bash
   pip install -r requirements.txt


## ProofFOL data
  The FOL dataset for ProofWriter (ProofFOL) is available on Hugging Face at:
👉 https://huggingface.co/datasets/ramyakeerthyt/ProofFOL

  You can load it using:
  ```
  from datasets import load_dataset
  dataset = load_dataset("ramyakeerthyt/ProofFOL")
  ```
## Training 
**Fine-tuning and Inference**
Fine-tuning and inference on three datasets (available in the ```data/``` folder) can be performed with the following commands:
```
python finetune.py
python inference.py
```
## Incremental Training
Incremental fine-tuning uses datasets located in ```data/incremental/```, created by systematically splitting the records to simulate a data augmentation scenario.

- **Fine-tuning:** Same process as above (```finetune.py```), but the dataset is structured for sequential learning.

- **Inference without a tool:** Run
  ```
  python inference_incremental.py
  ```
- **Inference with a tool:** Run
  ```
  python inference_t5.py
  ```

Additionally, you can train your own T5 models using:
```
python run_seq2seq.py
```
with the provided perturbation datasets.

## Licence
This code is licensed under the MIT License and is available for research purposes.

## Citation
If you use this repository, please cite:
> Thatikonda, R. K., Han, J., Buntine, W., & Shareghi, E. (2024). Strategies for improving nl-to-fol translation with llms: Data generation, incremental fine-tuning, and verification. arXiv preprint arXiv:2409.16461.
