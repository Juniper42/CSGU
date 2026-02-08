import time

import torch

from evaluate import evaluate
from logger import get_logger

logger = get_logger('Train')

def train(model, optimizer, link_data, args, device):
    best_f1_macro = 0
    test_info = {}
    patience = args.patience

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % args.eval_step == 0:
            eval_info = evaluate(model, link_data, device, eval_flag='test')
            t = time.time() - t
            logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'AUC: {eval_info["auc"]:.4f}, F1: {eval_info["f1"]:.4f}, MacroF1: {eval_info["f1_macro"]:.4f}, MicroF1: {eval_info["f1_micro"]:.4f}')
            if eval_info['f1_macro'] > best_f1_macro:
                best_f1_macro = eval_info['f1_macro']
                test_info = evaluate(model, link_data, device, eval_flag='test')
                test_info['epoch'] = epoch
                patience = args.patience
                # torch.save(model.state_dict(), f'models/{args.model}/id{args.in_dim}_od_{args.out_dim}_layer{args.layer_num}_sd{args.seed}_{args.dataset}_lr{args.lr}_wd{args.weight_decay}.pth')
                logger.info(f'Test Result: Epoch: {test_info["epoch"]:03d}, Loss: {loss:.4f}, '
                    f'AUC: {test_info["auc"]:.4f}, F1: {test_info["f1"]:.4f}, MacroF1: {test_info["f1_macro"]:.4f}, MicroF1: {test_info["f1_micro"]:.4f}')
            else:
                patience -= 1
            if patience <= 0:
                break
            
    return model