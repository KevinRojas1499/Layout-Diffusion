"""Evaluate a single fixed-length checkpoint with the (fixed) sampler.

Loads EMA weights, runs oracle sampling using lengths from --eval_data_path,
and prints accuracy / validity / MAE.
"""
import json
import click
import torch
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset
from utils.tokenizer import VocabTokenizer
from models.mmdit_qm9 import MMDiTFixedLength
from fixed_length_multimodal_interpolant import FixedLengthMultimodalInterpolant
from fixed_length_parenthesis_training import (
    collect_length_pairs,
    oracle_sampling,
    evaluate_samples,
    save_samples_jsonl,
)


@click.command()
@click.option('--checkpoint', required=True, type=str)
@click.option('--data_path', default='data/parenthesis/train_l5.jsonl', type=str,
              help='Train dataset (only used to get max_length)')
@click.option('--eval_data_path', default='data/parenthesis/test_l5.jsonl', type=str)
@click.option('--eval_samples', default=500, type=int)
@click.option('--eval_steps', default=50, type=int)
@click.option('--depth', default=8, type=int)
@click.option('--hidden_dim', default=256, type=int)
@click.option('--out_jsonl', default=None, type=str)
def main(checkpoint, data_path, eval_data_path, eval_samples, eval_steps, depth, hidden_dim, out_jsonl):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    euclidean_dim = 1

    character_tokenizer = VocabTokenizer(vocab={'+', '-', '*', '=', '(', ')'})
    train_dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=data_path)
    eval_dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=eval_data_path)
    eval_length_pairs = collect_length_pairs(eval_dataset)

    model = MMDiTFixedLength(
        euclidean_dim=euclidean_dim,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=depth,
        positions_depth=depth,
        depth=depth,
        dim_modalities=[hidden_dim, hidden_dim],
        dim_joint_attn=hidden_dim,
        dim_conds=[hidden_dim, hidden_dim],
    ).to(device)

    snap = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(snap['ema'], strict=False)
    model.eval()

    interpolant = FixedLengthMultimodalInterpolant(
        max_length=train_dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id,
        bos_token=character_tokenizer.bos_token_id,
        euclidean_dim=euclidean_dim,
    )

    samples = oracle_sampling(
        interpolant=interpolant,
        model=model,
        steps=eval_steps,
        eval_samples=eval_samples,
        length_pairs=eval_length_pairs,
        device=device,
    )

    metrics = evaluate_samples(samples, character_tokenizer, euclidean_dim=euclidean_dim, delta=0.5)
    validity = metrics['valid'] / metrics['total'] if metrics['total'] else 0.0
    print(f"\ncheckpoint: {checkpoint}")
    print(f"eval_samples={eval_samples}  eval_steps={eval_steps}")
    print(f"  accuracy        = {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['valid']})")
    print(f"  validity        = {validity:.4f}  ({metrics['valid']}/{metrics['total']})")
    print(f"  invalid         = {metrics['invalid']}")
    print(f"  mean_abs_error  = {metrics['mean_abs_error']:.4f}")

    if out_jsonl:
        save_samples_jsonl(samples, character_tokenizer, out_jsonl, euclidean_dim=euclidean_dim)
        print(f"  samples written to {out_jsonl}")


if __name__ == '__main__':
    main()
