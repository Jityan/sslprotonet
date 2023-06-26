import argparse
import datetime
import os.path as osp
import torch
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import resnet12
from utils import set_gpu, Averager, count_acc, euclidean_metric, seed_torch, compute_confidence_interval


def final_evaluate(args):
        if args.dataset == 'mini':
            valset = MiniImageNet('test', args.size)
        elif args.dataset == 'tiered':
            valset = TieredImageNet('test', args.size)
        elif args.dataset == "cifarfs":
            valset = CIFAR_FS('test', args.size)
        else:
            print("Invalid dataset...")
            exit()
        val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                        args.test_way, args.shot + args.test_query)
        loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=args.worker, pin_memory=True)

        if args.model == 'convnet':
            model = Convnet().cuda()
            print("=> Convnet architecture...")
        else:
            if args.dataset in ['mini', 'tiered']:
                model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
            else:
                model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
            print("=> Resnet architecture...")

        model.load_state_dict(torch.load(osp.join(args.save_path, 'max-acc.pth')))
        print("=> Model loaded...")
        model.eval()

        ave_acc = Averager()
        acc_list = []

        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.test_way * args.shot
            data_shot, data_query = data[:k], data[k:]

            x = model(data_shot)
            x = x.reshape(args.shot, args.test_way, -1).mean(dim=0)
            p = x

            logits = euclidean_metric(model(data_query), p)

            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            acc = count_acc(logits, label)
            ave_acc.add(acc)
            acc_list.append(acc*100)

            x = None; p = None; logits = None

        a, b = compute_confidence_interval(acc_list)
        print("Final accuracy with 95% interval : {:.2f}±{:.2f}".format(a, b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--model', type=str, default='convnet', choices=['convnet', 'resnet'])
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini','tiered','cifarfs'])
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    # fix seed
    seed_torch(1)
    set_gpu(args.gpu)

    if args.dataset in ['mini', 'tiered']:
        args.size = 84
    elif args.dataset in ['cifarfs']:
        args.size = 32
        args.worker = 0
    else:
        args.size = 28

    final_evaluate(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)