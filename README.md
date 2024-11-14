# Value Alignment Project

This project aims to align the responses of a GPT-2 model with specific ethical and moral values using a GAN-inspired approach. The project involves evaluating the baseline model, fine-tuning it using LoRA (Low-Rank Adaptation), and comparing the results to ensure value alignment.

## Project Structure

- `part1.ipynb`: Notebook for first step of training target model
- `part2.ipynb`: Notebook for GAN setup between 2 target models with unfrozen weights
- `README.md`: Project documentation.

## Installation

To run this project, you need to install the following dependencies using Poetry:

1. **Transformers**: A library for state-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.
2. **Torch**: An open-source machine learning library.
3. **Sage**: A library for scoring and evaluating model responses.
4. **LoRA**: A library for Low-Rank Adaptation of Large Language Models.

### Steps to Install

1. **Install Poetry**: If you haven't installed Poetry yet, you can do so by following the instructions [here](https://python-poetry.org/docs/#installation).

2. **Create a New Poetry Project**: Navigate to your project directory and create a new Poetry project.

```bash
poetry install

git clone --recurse-submodule https://github.com/HiIAmTzeKean/SC4001-Value-Alignment-Network
cd SaGE/library
pip install -e .
```