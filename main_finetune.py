import argparse
import json
import os
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import load_mimic_iii_mortality, load_mimic_iii_phenotyping, load_mimic_iii_decompensation, load_mimic_iii_lengthofstay
from data.dataloader import collate_fn
from models.smart import Encoder, Classifier
from utils.metrics import print_metrics_binary, print_metrics_multilabel, print_metrics_regression
from utils.utils import set_seed, distributed_init, init_logging


def test(args, checkpoint_path, test_dataloader):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path))
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    test_loss = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            h = encoder(**batch)
            preds = classifier(h, **batch)
            test_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
            preds_all.append(preds.cpu())
            labels_all.append(batch['labels'].cpu())
    print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_decompensation', choices=['c12', 'c19', 'mimic_mortality', 
                            'mimic_phenotyping', 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--data_dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='./export/')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.dataset, 'smart')
    if args.local_rank == 0 and args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.local_rank == 0:
        logger = logging.getLogger()
        init_logging(logger, args.save_dir if args.save_model else None)
    else:
        logger = None
    log(logger, json.dumps(vars(args), indent=4))
    set_seed(args.seed)

    if args.dataset == 'c12':
        args.input_dim = 37
        args.demo_dim = 4
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_challenge_2012()
    elif args.dataset == 'c19':
        args.input_dim = 34
        args.demo_dim = 5
        args.num_class = 2
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_challenge_2019()
    elif args.dataset == 'mimic_mortality':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_mimic_iii_mortality()
    elif args.dataset == 'mimic_phenotyping':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 25
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_mimic_iii_phenotyping()
    elif args.dataset == 'mimic_decompensation':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 24
        train_dataset, val_dataset, test_dataset = load_mimic_iii_decompensation()
    elif args.dataset == 'mimic_lengthofstay':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 1
        args.max_len = 24
        train_dataset, val_dataset, test_dataset = load_mimic_iii_lengthofstay()
    else:
        raise Exception("Dataset not exist!")
    if args.data_dropout > 0:
        train_dataset.dropout_data(args.data_dropout)
        val_dataset.dropout_data(args.data_dropout)
        test_dataset.dropout_data(args.data_dropout)
    log(logger, 'Dataset Loaded.')
    distributed_init(args)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=collate_fn)
    
    encoder = Encoder(args).cuda()
    classifier = Classifier(args).cuda()
    
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': classifier.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    if args.dataset == 'mimic_phenotyping':
        criterion = torch.nn.BCEWithLogitsLoss()
        print_metrics = print_metrics_multilabel
        save_metric = 'auc_macro'
    elif args.dataset == 'mimic_lengthofstay':
        criterion = torch.nn.MSELoss()
        print_metrics = print_metrics_regression
        save_metric = 'mse'
    else:
        print_metrics = print_metrics_binary
        save_metric = 'auprc'
    
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint-mse.pth'))
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])

    best_auc = 0
    best_prc = 0
    best_mse = 100
    for i in range(1, args.epochs + 1):
        train_loss = 0
        val_loss = 0
        encoder.train()
        classifier.train()
        for step, batch in enumerate(train_dataloader, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            if i <= args.freeze_epochs:
                with torch.no_grad():
                    h = encoder(**batch)
            else:
                h = encoder(**batch)
            preds = classifier(h, **batch)
            loss = criterion(preds, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['x'].shape[0]

        encoder.eval()
        classifier.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                h = encoder(**batch)
                preds = classifier(h, **batch)
                val_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
                preds_all.append(preds.cpu())
                labels_all.append(batch['labels'].cpu())
        metrics = print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, train_loss / len(train_dataset) * args.world_size, val_loss / len(val_dataset)))
        cur_mse = val_loss / len(val_dataset)
        if save_metric != 'mse':
            if metrics[save_metric] > best_prc:
                best_prc = metrics[save_metric]
                if args.local_rank == 0:
                    state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'epoch': i
                    }
                    log(logger, f'----- Save best model - {save_metric}: %.4f -----' % metrics[save_metric])
                    torch.save(state, os.path.join(args.save_dir, 'checkpoint-prc.pth'))
        else:
            if metrics[save_metric] < best_mse:
                best_mse = metrics[save_metric]
                if args.local_rank == 0:
                    state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'epoch': i
                    }
                    log(logger, f'----- Save best model - {save_metric}: %.4f -----' % metrics[save_metric])
                    torch.save(state, os.path.join(args.save_dir, 'checkpoint-prc.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-prc.pth', test_dataloader)
