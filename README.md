# Using LLM to Generate Code for Classification and Regression Tasks

## Quick Start

```shell
# Clone the repository
git clone https://github.com/zsworld6/ML-hw.git
cd ML-hw

# Install dependencies
conda env create -f environment.yml
conda activate run_env

# Run the code
./src/scripts/run.sh
```

## Introduction

### Problem Statement

Given a dataset, we want to use large language models(LLMs) to generate code that can be used to train a machine learning model on the dataset. The generated code should be able to handle classification or regression tasks.

### Pipeline

1. model: We use different large language models to generate code for the given dataset. Some models have been supervised fine-tuned on code generation tasks, while others are unsupervised models.
2. code generation: We use the model to generate code for the given dataset. The generated code is in the form of a python script.
3. evaluation: We evaluate the generated code by running it on the dataset. If the code can correctly run, we use the accuracy or mean squared error of the generated code as the evaluation metric. Otherwise, we use the error message as the evaluation metric.
4. reflection: We give the generated code and the evaluation results to the LLM and let it self-reflect. The LLM will generate a reflection report on the generated code.
5. iteration: We give the reflection report to the LLM and let it generate new code. We repeat the process until the generated code can correctly run on the dataset or get a good evaluation metric.

### Details

#### Reflexion

See [Reflexion](https://github.com/noahshinn/reflexion).

Inspired by this idea, we use the LLM to generate a reflection report on the generated code. The reflection report contains the evaluation results and some suggestions for improving the generated code. Then the model can self-debug.

#### CAAFE

See [CAAFE](https://github.com/automl/CAAFE/tree/main).

We tried to let the LLM use the CAAFE to get a better performance by teaching it in the prompts. Some examples of the prompts can see in the [`templates.py`](./src/caafe/templates.py).

#### Supervised Fine-tuning

We fine-tuned the LLM on the code generation tasks. The fine-tuned models can generate code that is more likely to be correct.

We use the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) framework to fine-tune the LLM.

Datasets used for fine-tuning:

- [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

- [AlpacaCode](https://huggingface.co/datasets/mwitiderrick/AlpacaCode)