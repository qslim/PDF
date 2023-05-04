import os
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from tqdm import tqdm
### importing OGB
from ogb.graphproppred import Evaluator, collate_dgl

import csv

import sys
sys.path.append('../..')
from ogbg.data_preparation import DglGraphPropPred
from model import Net
from utils.config import process_config, get_args
from utils.lr import warm_up_lr

torch.set_num_threads(1)

def train(model, device, loader, optimizer, criterion):
    model.train()
    loss_all = 0

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Train iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        if x.shape[0] == 1:
            pass
        else:
            pred = model(bg, x, edge_attr, bases)
            optimizer.zero_grad()

            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = labels == labels
            loss = criterion(pred.to(torch.float32)[is_labeled], labels.to(torch.float32)[is_labeled])
            loss.backward()
            loss_all = loss_all + loss.detach().item()
            optimizer.step()
    return loss_all / len(loader)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Eval iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop('feat')
        edge_attr = bg.edata.pop('feat')
        bases = bg.edata.pop('bases')
        labels = labels.to(device)

        if x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(bg, x, edge_attr, bases)

            y_true.append(labels.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


import time
def main():
    args = get_args()
    config = process_config(args)
    cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(torch.cuda.get_device_name(0), cuda_id)
    print(config)

    algo_setting = str(config.commit_id[0:7]) + '_' + str(cuda_id) \
                   + str(config.get('shared_filter', '')) \
                   + str(config.get('linear_filter', '')) \
                   + str(config.basis) \
                   + 'E' + str(config.epsilon) \
                   + 'P' + str(config.power) \
                   + 'D' + str(config.get('degs', '')) + '_' \
                   + str(config.architecture.layers) + '_' \
                   + str(config.architecture.hidden) + '_' \
                   + str(config.hyperparams.learning_rate) + '_' \
                   + str(config.hyperparams.warmup_epochs) \
                   + str(config.hyperparams.milestones) \
                   + str(config.hyperparams.decay_rate) + '_' \
                   + str(config.hyperparams.weight_decay) \
                   + 'B' + str(config.hyperparams.batch_size) \
                   + 'W' + str(config.get('num_workers', 'na'))

    algo_setting = algo_setting.replace(' ', '').replace('[', ':').replace(']', ':')
    csv_dir = config.directory + 'stat/'

    os.makedirs(os.path.dirname(csv_dir + algo_setting + '/'), exist_ok=True)
    path_stat_total = csv_dir + algo_setting + '/' + str(config.time_stamp) + 'stat_total.csv'
    with open(path_stat_total, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['ts_fk_algo_hp', 'seed', 'test', 'valid',
                             'best_val_epoch', 'best_train', 'min_train_loss'])
        csv_file.flush()

    for seed in config.seeds:
        config.seed = seed
        config.time_stamp = int(time.time())
        print(config)
        ts_fk_algo_hp = algo_setting + '/T' + str(config.time_stamp) + '_S' + str(config.seed)

        epoch_idx, train_curve, valid_curve, test_curve, trainL_curve, task_type = run_with_given_seed(config, ts_fk_algo_hp)

        with open(csv_dir + ts_fk_algo_hp + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train', 'valid', 'test', 'train_loss'])
            csv_writer.writerows(
                np.transpose(np.array([epoch_idx, train_curve, valid_curve, test_curve, trainL_curve])))
            csv_file.flush()

        if 'classification' in task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_curve)
        print('Finished test: {}, Validation: {}, epoch: {}, best train: {}, best loss: {}'
              .format(test_curve[best_val_epoch], valid_curve[best_val_epoch],
                      best_val_epoch, best_train, min(trainL_curve)))

        with open(path_stat_total, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([ts_fk_algo_hp, config.seed, test_curve[best_val_epoch], valid_curve[best_val_epoch],
                                 best_val_epoch, best_train, min(trainL_curve)])
            csv_file.flush()

    with open(path_stat_total, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        column_test = []
        column_valid = []
        for row in csv_reader:
            column_test.append(row['test'])
            column_valid.append(row['valid'])

        column_test = np.array(column_test, dtype=float)
        column_valid = np.array(column_valid, dtype=float)
        test_stat = str(np.mean(column_test)) + '_' + str(np.std(column_test))
        valid_stat = str(np.mean(column_valid)) + '_' + str(np.std(column_valid))

    with open(path_stat_total, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['', '', test_stat, valid_stat, '', '', ''])
        csv_file.flush()


def run_with_given_seed(config, ts_fk_algo_hp):
    if config.get('seed') is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### automatic dataloading and splitting
    dataset = DglGraphPropPred(name=config.dataset_name)

    print("Bases total: {}".format(dataset.graphs[0].edata['bases'].shape[1]))

    split_idx = dataset.get_idx_split()
    # train_idx = filter_train_set(split_idx["train"], dataset)

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(config.dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config.hyperparams.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config.hyperparams.batch_size, shuffle=False,
                              num_workers=config.num_workers, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=config.hyperparams.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=collate_dgl)

    model = Net(config.architecture, num_tasks=dataset.num_tasks,
                num_basis=dataset.graphs[0].edata['bases'].shape[1]).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.AdamW(model.parameters(), lr=config.hyperparams.learning_rate,
                            weight_decay=config.hyperparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.hyperparams.milestones,
                                                     gamma=config.hyperparams.decay_rate)

    if "classification" in dataset.task_type:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss()

    epoch_idx = []
    valid_curve = []
    test_curve = []
    train_curve = []
    trainL_curve = []

    writer = SummaryWriter(config.directory + 'board/')

    cur_epoch = 0
    # if config.get('resume_train') is not None:
    #     print("Loading model from {}...".format(config.resume_train), end=' ')
    #     checkpoint = torch.load(config.resume_train)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(device)
    #     criterion.load_state_dict(checkpoint['criterion_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     cur_epoch = checkpoint['epoch']
    #     cur_loss = checkpoint['loss']
    #     lr = checkpoint['lr']
    #     print("Model loaded.")
    #
    #     print("Epoch {} evaluating...".format(cur_epoch))
    #     train_perf = eval(model, device, train_loader, evaluator)
    #     valid_perf = eval(model, device, valid_loader, evaluator)
    #     test_perf = eval(model, device, test_loader, evaluator)
    #
    #     print('Train:', train_perf[dataset.eval_metric],
    #           'Validation:', valid_perf[dataset.eval_metric],
    #           'Test:', test_perf[dataset.eval_metric],
    #           'Train loss:', cur_loss,
    #           'lr:', lr)
    #
    #     train_curve.append(train_perf[dataset.eval_metric])
    #     valid_curve.append(valid_perf[dataset.eval_metric])
    #     test_curve.append(test_perf[dataset.eval_metric])
    #     trainL_curve.append(cur_loss)
    #
    #     writer.add_scalars('traP', {ts_fk_algo_hp: train_perf[dataset.eval_metric]}, cur_epoch)
    #     writer.add_scalars('valP', {ts_fk_algo_hp: valid_perf[dataset.eval_metric]}, cur_epoch)
    #     writer.add_scalars('tstP', {ts_fk_algo_hp: test_perf[dataset.eval_metric]}, cur_epoch)
    #     writer.add_scalars('traL', {ts_fk_algo_hp: cur_loss}, cur_epoch)
    #     writer.add_scalars('lr',   {ts_fk_algo_hp: lr}, cur_epoch)

    best_val = 0.0
    for epoch in range(cur_epoch + 1, config.hyperparams.epochs + 1):
        if epoch <= config.hyperparams.warmup_epochs:
            warm_up_lr(epoch, config.hyperparams.warmup_epochs, config.hyperparams.learning_rate, optimizer)
        lr = scheduler.optimizer.param_groups[0]['lr']
        # print("Epoch {} training...".format(epoch))
        train_loss = train(model, device, train_loader, optimizer, criterion)
        if epoch > config.hyperparams.warmup_epochs:
            scheduler.step()
        # scheduler.step()

        # print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print('Epoch:', epoch,
              'Train:', train_perf[dataset.eval_metric],
              'Validation:', valid_perf[dataset.eval_metric],
              'Test:', test_perf[dataset.eval_metric],
              'Train loss:', train_loss,
              'lr:', lr)

        epoch_idx.append(epoch)
        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        trainL_curve.append(train_loss)

        writer.add_scalars('traP', {ts_fk_algo_hp: train_perf[dataset.eval_metric]}, epoch)
        writer.add_scalars('valP', {ts_fk_algo_hp: valid_perf[dataset.eval_metric]}, epoch)
        writer.add_scalars('tstP', {ts_fk_algo_hp: test_perf[dataset.eval_metric]}, epoch)
        writer.add_scalars('traL', {ts_fk_algo_hp: train_loss}, epoch)
        writer.add_scalars('lr',   {ts_fk_algo_hp: lr}, epoch)

        if config.get('checkpoint_dir') is not None:
            filename_header = str(config.commit_id[0:7]) + '_' \
                       + str(config.time_stamp) + '_' \
                       + str(config.dataset_name)
            if valid_perf[dataset.eval_metric] > best_val:
                best_val = valid_perf[dataset.eval_metric]
                filename = filename_header + 'best.tar'
            else:
                filename = filename_header + 'curr.tar'

            print("Saving model as {}...".format(filename), end=' ')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'criterion_state_dict': criterion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': train_loss,
                        'lr': lr},
                       os.path.join(config.checkpoint_dir, filename))
            print("Model saved.")

    writer.close()

    return epoch_idx, train_curve, valid_curve, test_curve, trainL_curve, dataset.task_type


if __name__ == "__main__":
    main()
