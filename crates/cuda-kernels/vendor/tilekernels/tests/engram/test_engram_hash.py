import os
import pytest
import torch

from tile_kernels.engram import engram_hash
from tile_kernels.torch.engram import engram_hash_ref, make_offsets
from tile_kernels.testing.generator import generate_num_tokens
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_tokens = params['num_tokens']
    max_ngram_size = params['ngram']
    num_ngram_layers = params['layers']
    num_embed_table_per_ngram = params['tables']
    ngram_token_ids = torch.randint(0, 100000, (num_tokens, max_ngram_size), dtype=torch.int32, device='cuda')
    multipliers = torch.randint(0, 100000, (num_ngram_layers, max_ngram_size), dtype=torch.int64, device='cuda')
    vocab_sizes = torch.randint(100000, 1000000, (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), dtype=torch.int32, device='cuda')
    offsets = make_offsets(vocab_sizes)
    return (ngram_token_ids, multipliers, vocab_sizes, offsets)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    return [
        {'num_tokens': t}
        for t in generate_num_tokens(is_benchmark=is_benchmark)
    ]


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_engram_hash(params):
    num_tokens = params['num_tokens']
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8

    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(
        {'num_tokens': num_tokens, 'ngram': max_ngram_size, 'layers': num_ngram_layers, 'tables': num_embed_table_per_ngram})

    # Correctness
    output = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)
    output_ref = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)
    assert_equal(output, output_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_engram_hash_benchmark(benchmark_timer, benchmark_record, params):
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8

    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(
        {**params, 'ngram': max_ngram_size, 'layers': num_ngram_layers, 'tables': num_embed_table_per_ngram})
    output = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)

    t_us = benchmark_timer(lambda: engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets))

    num_bytes = count_bytes(ngram_token_ids, multipliers, vocab_sizes, offsets, output)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_hash',
        operation='fwd',
        params={**params, 'ngram': max_ngram_size, 'layers': num_ngram_layers, 'tables': num_embed_table_per_ngram},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
