import argparse
import datetime
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.samplers import CategoriesSampler
from models.convnet import Convnet
from models.distill import DistillKL, HintLoss
from models.resnet import resnet12
from utils import set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, seed_torch, compute_confidence_interval


def get_dataset(args):
    if args.dataset == 'mini':
        trainset = MiniImageNet('train', args.size)
        valset = MiniImageNet('test', args.size)
        print("=> MiniImageNet...")
    elif args.dataset == 'tiered':
        trainset = TieredImageNet('train', args.size)
        valset = TieredImageNet('test', args.size)
        print("=> TieredImageNet...")
    elif args.dataset == 'cifarfs':
        trainset = CIFAR_FS('train', args.size)
        valset = CIFAR_FS('test', args.size)
        print("=> CIFAR FS...")
    else:
        print("Invalid dataset...")
        exit()
    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                        args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=args.worker, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)
    return train_loader, val_loader

def training(args):
        ensure_path(args.save_path)

        train_loader, val_loader = get_dataset(args)

        if args.model == 'convnet':
            teacher = Convnet().cuda()
            print("=> Convnet architecture...")
        else:
            if args.dataset in ['mini', 'tiered']:
                teacher = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5).cuda()
            else:
                teacher = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2).cuda()
            print("=> Resnet architecture...")

        if args.kd_mode != 0:
            # produce a student model with the same structure as teacher model without knowldege
            model = copy.deepcopy(teacher)
            if args.stage1_path:
                model.load_state_dict(torch.load(osp.join(args.stage1_path, 'max-acc.pth')))
                print("=> Student loaded with pretrain knowledge...")

        teacher.load_state_dict(torch.load(osp.join(args.stage2_path, 'max-acc.pth')))
        print("=> Teacher model loaded...")

        if args.kd_mode == 0:
            # intilialize student with same knowledge as teacher
            model = copy.deepcopy(teacher)
            print("=> Student obtain teacher's knowledge...")

        if args.kd_type == 'kd':
            criterion_kd = DistillKL(args.temperature).cuda()
        else:
            criterion_kd = HintLoss().cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

        def save_model(name):
            torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
        
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0

        timer = Timer()
        best_epoch = 0
        cmi = [0.0, 0.0]

        for epoch in range(1, args.max_epoch + 1):

            tl, ta = train(args, teacher, model, train_loader, optimizer, criterion_kd)
            lr_scheduler.step()
            vl, va, aa, bb = validate(args, model, val_loader)

            if va > trlog['max_acc']:
                trlog['max_acc'] = va
                save_model('max-acc')
                best_epoch = epoch
                cmi[0] = aa
                cmi[1] = bb

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            torch.save(trlog, osp.join(args.save_path, 'trlog'))

            save_model('epoch-last')
            ot, ots = timer.measure()
            tt, _ = timer.measure(epoch / args.max_epoch)

            print('Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f} - max acc={:.4f} - ETA:{}/{}'.format(
                epoch, args.max_epoch, tl, ta, vl, va, trlog['max_acc'], ots, timer.tts(tt-ot)))
            
            if epoch == args.max_epoch:
                print("Best Epoch is {} with acc={:.2f}±{:.2f}%...".format(best_epoch, cmi[0], cmi[1]))
                print("---------------------------------------------------")

def ssl_loss(args, model, data_shot):
    # s1 s2 q1 q2 q1 q2
    x_90 = data_shot.transpose(2,3).flip(2)
    x_180 = data_shot.flip(2).flip(3)
    x_270 = data_shot.flip(2).transpose(2,3)
    data_query = torch.cat((x_90, x_180, x_270),0)

    proto = model(data_shot)
    proto = proto.reshape(1, args.shot*args.train_way, -1).mean(dim=0)

    label = torch.arange(args.train_way * args.shot).repeat(args.pre_query)
    label = label.type(torch.cuda.LongTensor)

    logits = euclidean_metric(model(data_query), proto)
    loss = F.cross_entropy(logits, label)

    return loss

def train(args, teacher, model, train_loader, optimizer, criterion_kd):
        teacher.eval()
        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:] # datashot (30, 3, 84, 84)

            # teacher
            with torch.no_grad():
                tproto = teacher(data_shot)
                ft = tproto
                ft = [f.detach() for f in ft]
                tproto = tproto.reshape(args.shot, args.train_way, -1).mean(dim=0)
                # soft target from teacher
                tlogits = euclidean_metric(teacher(data_query), tproto)

            proto = model(data_shot) # (30, 1600)
            fs = proto
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.train_query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            acc = count_acc(logits, label)

            if args.kd_mode != 0:
                # few-shot loss from student
                clsloss = F.cross_entropy(logits, label)

            # distillation loss
            if args.kd_type == 'kd':
                kdloss = criterion_kd(logits, tlogits)
            else:
                kdloss = criterion_kd(fs[-1], ft[-1])
            
            # self-supervised loss signal
            loss_ss = ssl_loss(args, model, data_shot)

            if args.kd_mode != 0:
                loss = ((1.0 - args.kd_coef) * clsloss) + (args.kd_coef * kdloss) + (args.ssl_coef * loss_ss)
            else:
                loss = kdloss + (args.ssl_coef * loss_ss)

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

        return tl.item(), ta.item()

def validate(args, model, val_loader):
        model.eval()

        vl = Averager()
        va = Averager()
        acc_list = []

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            acc_list.append(acc*100)
            
            proto = None; logits = None; loss = None
        a,b = compute_confidence_interval(acc_list)
        return vl.item(), va.item(), a, b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--pre-query', type=int, default=3) # for self-supervised process: the number of query image generated based on support image
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--model', type=str, default='convnet', choices=['convnet', 'resnet'])
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini','tiered','cifarfs'])
    parser.add_argument('--ssl-coef', type=float, default=0.1, help='The beta coefficient for self-supervised loss')
    # self-distillation stage parameter
    parser.add_argument('--temperature', type=int, default=4)
    parser.add_argument('--kd-coef', type=float, default=0.1, help="The gamma coefficient for distillation loss")
    # 0: copy teacher and only KD       1: common KD
    parser.add_argument('--kd-mode', type=int, default=1, choices=[0,1])
    parser.add_argument('--kd-type', type=str, default='kd', choices=['kd', 'hint'])
    parser.add_argument('--stage1-path', default='')
    parser.add_argument('--stage2-path', default='')
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

    training(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)

