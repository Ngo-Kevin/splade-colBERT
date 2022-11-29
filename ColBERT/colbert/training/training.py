import time
import torch
import random
import torch.nn as nn
import numpy as np
import ujson
import os

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def filter_layers(name, prune_type, ignore_bias=True):
    if name.startswith('model.bert.embeddings') \
        or 'LayerNorm' in name: 
            return True
    if ignore_bias and name.endswith('bias'):
        return True
    if prune_type == "dense":
        if "attention" in name:
            return True
    elif "attention" in prune_type:
        if "attention" not in name:
            return True
        if "no_dense" in prune_type and "dense" in name:
            return True
    return False

def train(config: ColBERTConfig, triples, queries=None, collection=None):
    
    ### pruning
    ###### Gotta do full intergration w/ colBERT?
    prune_type = 'attention'
    prune_l1_lambda = 5e-3
    ###

    ### resume ###
    # checkpoint = {}
    # config.set('resume', True)
    # checkpoint['batch'] = 0
    # config.checkpoint = r'D:\Documents\Python_Scripts\class\UCB\splade-colBERT\ColBERT\experiments\msmarco_400.000_l1_mean_1e-5\2022-11\24\03.43.22\checkpoints\colbert-80000'
    ###
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    print("Using config.checkpoint =", config.checkpoint)
    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)
    
    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    if config.resume:
        assert config.checkpoint is not None
        start_batch_idx = checkpoint['batch']
        reader.skip_to_batch(start_batch_idx, 64)

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0
        l1_penality_sum = 0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
                
                if config.use_ib_negatives:
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss


                loss = loss / config.accumsteps

            # apply l1 regularization
            if prune_l1_lambda > 0:
                l1_norm = np.mean([p.abs().sum().cpu().detach().numpy() 
                    for name, p in colbert.named_parameters() 
                    if not filter_layers(name, prune_type)])
                l1_penality = prune_l1_lambda * l1_norm
                l1_penality_sum += l1_penality
                loss += l1_penality

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        ckpt_loss = {'training_loss': train_loss,
                    'l1_penality': l1_penality_sum,
                    'colbert_loss':train_loss-l1_penality_sum}
        if config.rank < 1:
            if l1_penality_sum > 0:
                print_message(batch_idx, train_loss-l1_penality_sum, l1_penality_sum)
            else:
                print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, save_extra_param=ckpt_loss)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True, save_extra_param=ckpt_loss)
        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.
    


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
