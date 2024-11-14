# -*- coding: utf-8 -*-
"""coling-edge-generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bCZSzXSKSFGJS32ok6q1rl9jON6YFoMX
"""

import argparse
import os
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from tqdm import tqdm


tqdm.pandas()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(
    'sentence-transformers/stsb-distilroberta-base-v2', device=device)


def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=SmoothingFunction().method1)


def calculate_rouge(reference, candidate):
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return rouge_scorer_obj.score(reference, candidate)['rougeL'].fmeasure


def calculate_bert(reference, candidate):
    _, _, bert_f1 = bert_score([candidate], [
                               reference], lang='en', model_type='bert-base-uncased', device=device)
    return bert_f1.item()


def calculate_cosine_similarity(reference, candidate):
    global model
    embeddings1 = model.encode([reference], convert_to_tensor=True)
    embeddings2 = model.encode([candidate], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings1, embeddings2)
    return float(cosine_sim[0][0])


def calculate_scores(row):
    scores = row.to_dict()
    scores['bleu_model_output'] = calculate_bleu(
        row['model_output_1'], row['model_output_2'])
    scores['rouge_model_output'] = calculate_rouge(
        row['model_output_1'], row['model_output_2'])
    # scores['bert_model_output'] = calculate_bert(
    #     row['model_output_1'], row['model_output_2'])
    scores['cosine_model_output'] = calculate_cosine_similarity(
        row['model_output_1'], row['model_output_2'])

    # scores['bleu_rot'] = calculate_bleu(row['rot_1'], row['rot_2'])
    # scores['rouge_rot'] = calculate_rouge(row['rot_1'], row['rot_2'])
    # scores['bert_rot'] = calculate_bert(row['rot_1'], row['rot_2'])
    # scores['cosine_rot'] = calculate_cosine_similarity(
    #     row['rot_1'], row['rot_2'])

    return pd.Series(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process input and output file paths.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    # Get the absolute paths for input and output files
    input_file_path = os.path.abspath(args.input_file)
    output_file_path = os.path.abspath(args.output_file)

    print("Input file path:", input_file_path)
    print("Output file path:", output_file_path)

    df = pd.read_csv(input_file_path)
    df = df.progress_apply(calculate_scores, axis=1)

    df.to_csv(output_file_path, index=False)
