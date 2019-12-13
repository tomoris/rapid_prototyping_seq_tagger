#!/usr/bin/env python3
# -*- coding;utf-8 -*-

import random
import argparse
import importlib

import torch
import torch.optim as optim

from utils.data_container import Data_container
from model.rapid_prototyping_seq_tagger import Rapid_prototyping_seq_tagger
from utils.eval_metrics import get_all_scores

from logging import getLogger
from utils.logger_config import load_logger_config
load_logger_config()
logger = getLogger(__name__)

logger.debug('Start logger')

torch.manual_seed(0)
random.seed(0)


def eval(model, data_container, batch_size, data_type, use_gpu, device):
    model.eval()
    all_correct_list = []
    all_predict_list = []
    with torch.no_grad():
        for i, batched_data in enumerate(
                data_container(data_type, batch_size)):
            # if use_gpu:
            # batched_data is (token_ids, char_ids, label_ids
            # multi_hot_label_tensor, masks)
            token_ids = batched_data[0].to(device)
            char_ids = batched_data[1].to(device)
            mask = batched_data[4].to(device)
            multi_hot_label_tensor = batched_data[3].to(device)
            tokens = batched_data[5]
            loss, predict_triplets_list = model(
                token_ids, char_ids, mask, multi_hot_label_tensor, calc_loss=True, sents=tokens)
            all_correct_list.extend(batched_data[2])
            all_predict_list.extend(predict_triplets_list)
    all_scores = get_all_scores(
        all_predict_list,
        all_correct_list,
        data_container.i2l,
        data_container.i2semil,
        data_container.semi_markov)
    return all_scores


def train(config_file):
    # load config file
    logger.info('Load config file')
    config_container = importlib.import_module(
        config_file.replace('.py', '').replace('/', '.')).get_config()
    logger.info('Load corpus')
    data_container = Data_container(config_container)

    logger.info('Building initial model')
    model = Rapid_prototyping_seq_tagger(config_container, data_container)
    device = torch.device('cpu')
    if config_container.use_gpu:
        device = torch.device('cuda')
        model = model.to(device)
    logger.info('Set up optimizer')
    optimizer = optim.SGD(
        model.parameters(),
        lr=config_container.lr,
        momentum=config_container.momentum,
        weight_decay=config_container.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=config_container.lr) #,
    # momentum=config_container.momentum,
    # weight_decay=config_container.weight_decay)

    dev_max_score = float('-inf')
    dev_max_flag = False
    final_scores = None

    logger.info('Start training')
    for epoch in range(config_container.epoch):
        lr = config_container.lr / (1.0 + (epoch * config_container.lr_decay))
        logger.debug('Change optimizer, lr = {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()
        total_loss = 0.0
        for i, batched_data in enumerate(data_container('train', config_container.batch_size)):
            if i % 100 == 0:
                logger.info('epoch:{0} training instance {1}'.format(epoch, i * config_container.batch_size))
            model.zero_grad()
            # if config_container.use_gpu:
            # batched_data is
            # (token_ids, char_ids, label_ids, multi_hot_label_tensor, masks, tokens)
            token_ids = batched_data[0].to(device)
            char_ids = batched_data[1].to(device)
            mask = batched_data[4].to(device)
            multi_hot_label_tensor = batched_data[3].to(device)
            tokens = batched_data[5]
            loss, predict_triplets_list = model(
                token_ids, char_ids, mask, multi_hot_label_tensor, sents=tokens, calc_loss=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            logger.debug('epoch:{0} i:{1} loss:{2}'.format(epoch, i, loss.item()))
        logger.debug('epoch:{0} total_loss:{1:.4f}'.format(epoch, total_loss))

        if config_container.use_PYHSMM:
            model.train_PYHSMM(data_container.data_container_for_PYHSMM, config_container.PYHSMM_threads,
                               config_container.PYHSMM_batch)
        if data_container.train_file_is_partially_labeled is False:
            all_scores = eval(model, data_container, config_container.batch_size,
                              'train', config_container.use_gpu, device)
            logger.debug('epoch:{0} train corpus F score = {1:.4f}'.format(epoch, all_scores['micro_f_1.0']))
        if config_container.dev_file is not None:
            all_scores = eval(model, data_container, config_container.batch_size,
                              'dev', config_container.use_gpu, device)
            logger.debug('epoch:{0} dev corpus F score = {1:.4f}'.format(epoch, all_scores['micro_f_1.0']))
            logger.debug('epoch:{0} dev corpus precision = {1:.4f}'.format(epoch, all_scores['micro_precision']))
            logger.debug('epoch:{0} dev corpus recall = {1:.4f}'.format(epoch, all_scores['micro_recall']))
            if dev_max_score < all_scores['micro_f_1.0']:
                dev_max_score = all_scores['micro_f_1.0']
                dev_max_flag = True
        if config_container.test_file is not None:
            all_scores = eval(model, data_container, config_container.batch_size,
                              'test', config_container.use_gpu, device)
            logger.debug('epoch:{0} test corpus F score = {1:.4f}'.format(epoch, all_scores['micro_f_1.0']))
            logger.debug('epoch:{0} test corpus precision = {1:.4f}'.format(epoch, all_scores['micro_precision']))
            logger.debug('epoch:{0} test corpus recall = {1:.4f}'.format(epoch, all_scores['micro_recall']))
            if dev_max_flag:
                final_scores = all_scores
                dev_max_flag = False
    logger.info('End training')

    if config_container.test_file is not None:
        if final_scores is None:
            final_scores = all_scores
        else:
            logger.info('dev corpus F score (max score duaring training epoch) = {0:.4f} '.format(dev_max_flag))
        logger.info('final test corpus F score = {1:.4f}'.format(epoch, all_scores['micro_f_1.0']))
        logger.info('final test corpus precision = {1:.4f}'.format(epoch, all_scores['micro_precision']))
        logger.info('final test corpus recall = {1:.4f}'.format(epoch, all_scores['micro_recall']))

        for key, item in all_scores.items():
            if isinstance(item, dict):
                for key_key, item_item in item.items():
                    logger.info(
                        'final test corpus eval metric {0} {1} \
                        = prec. {2:.4f}, recall {3:.4f}, f {3:.4f}'.format(
                            key, key_key, item_item[0],
                            item_item[1], item_item[2]))
            else:
                logger.info(
                    'final test corpus eval metric {0} = {1:.4f}'.format(
                        key, item))


def predict(config_file):
    pass


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--mode',
        help='train or predict',
        type=str,
        choices=['train', 'predict'],
        required=True)
    parser.add_argument(
        '--config',
        help='configuration file path',
        type=str,
        required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        logger.info('train mode')
        train(args.config)
    elif args.mode == 'predict':
        logger.info('predict mode')
        raise NotImplementedError()

    logger.debug('End main funcion')
    return


if __name__ == "__main__":
    main()
