## Setup

```
conda create -n ENV_NAME python==3.10.13 -y && conda activate ENV_NAME
conda install nvidia/label/cuda-12.1.1::cuda-toolkit -y
conda install pytorch==2.2.1 torchvision==0.17.1 -c pytorch -y
pip install -r requirements.txt
```

Exllamav2 install:

```
git clone https://github.com/turboderp/exllamav2 && cd exllamav2
git checkout v0.0.15
pip install -e .
```

Download exl2 gemma weights:

```
git lfs install
git clone https://huggingface.co/turboderp/Gemma-7B-it-exl2 && cd Gemma-7B-it-exl2
git checkout 6.0bpw
```

> Note: Install `git lfs`: https://git-lfs.com/



## Generate data

### Generate free prompts using GPT fewshot

```
python -m src.generation.gpt_free_prompts
```

- dataset: https://www.kaggle.com/datasets/tomirol/gpt-free-prompts


### Generate conditioned prompts using GPT fewshot

```
python -m src.generation.gpt_conditioned_prompts
```

- dataset: https://www.kaggle.com/datasets/tomirol/gpt-conditioned-prompts


### Generate rewritten text using exllamav2

```
python -m src.generation.gemma_rewritten_text_exllama.py
```

- dataset: https://www.kaggle.com/datasets/tomirol/gemma-rewritten-text-exllama


## Train for subject prediction

Create dataset:

```
python -m src.train.create_dataset
```

Train:

```
python -m src.train.mistral
```
