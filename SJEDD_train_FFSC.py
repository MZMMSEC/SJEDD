import pdb

import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import clip
import random, json, time, os, argparse, datetime
import numpy as np
from timm.utils import AverageMeter
from logger import create_logger
from sklearn.metrics import roc_auc_score as AUC
from utils import save_checkpoint, convert_models_to_fp32, load_checkpoint

from set_dataset_train import set_dataset_singleGPU, stack_label
from config_train import get_config
from lambda_optimizer import AutoLambda_SO, create_task_flags_CLIP, get_weight_str

from SO_Loss import pLoss_all_fidelity
from SO_Graph import graph_SA_ffso


def parse_option():
    parser = argparse.ArgumentParser('SJEDD_training', add_help=False)
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--auc', action='store_true')

    parser.add_argument('--initial_lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_epoch', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--opt', type=int, default=0)
    parser.add_argument('--txt_path_train', type=str, default="/data0/mian2/vision-language/data_build/train.txt")
    parser.add_argument('--txt_path_val', type=str, default="/data0/mian2/vision-language/data_build/val.txt")

    parser.add_argument('--name', default='CLIP_DeepFake_Abalation', type=str)
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test_log', type=str, default='')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--aug_probs', type=float, default=0.3)

    parser.add_argument('--VM', type=str, default="ViT-B/32", help='RN50, RN101, ViT-B/16, ViT-L/14')

    # for model.train or eval
    parser.add_argument('--is_model_train', action='store_true')
    #
    parser.add_argument('--weighting_method', default='auto-l', type=str)
    parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
    parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
    parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
    parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
    parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')
    parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
    parser.add_argument('--lambda_epoch', default=1, type=int)

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config, args):
    # ---------------------------------
    # define the templates
    # ---------------------------------
    l1_level = ['fake']

    l2_level = ['age', 'expression', 'gender', 'identity', 'pose']

    l3_level = ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin']

    l1_texts = torch.cat([clip.tokenize(f"A photo of a {l1} face") for l1 in l1_level]).to('cuda').unsqueeze(0)

    l2_texts = torch.cat(
        [clip.tokenize(f"A photo of a face with the global attribute of {l2} altered") for l2 in l2_level]).to(
        'cuda').unsqueeze(0)

    l3_texts = torch.cat(
        [clip.tokenize(f"A photo of a face with the local attribute of {l3} altered") for l3 in l3_level]).to(
        'cuda').unsqueeze(0)

    joint_texts = [l1_texts, l2_texts, l3_texts]

    # -----------------------------------
    # ablations: define the visual model
    # -----------------------------------
    mode_name = args.VM # default setting is "ViT-B/32"
    model, preprocess, _, __ = clip.load(mode_name, device='cuda', jit=False)
    model.cuda()
    model_without_ddp = model

    # --------
    # logging the tasks
    # --------
    train_tasks = create_task_flags_CLIP('all')
    pri_tasks = create_task_flags_CLIP(args.task)
    train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
    pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
    logger.info('Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode '
                .format(train_tasks_str, pri_tasks_str))
    logger.info('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
                .format(args.weight.title(), args.grad_method.upper()))

    # -----------------------------------
    # define the loss weighting
    # -----------------------------------
    total_epoch = args.num_epoch
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    params = model.parameters()

    autol = AutoLambda_SO(model_without_ddp, device, train_tasks, pri_tasks, args.autol_init, dataset='ffsc')
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = torch.optim.Adam([autol.meta_weights], lr=args.autol_lr)

    _, data_loader_val_autoL = set_dataset_singleGPU(args.txt_path_train, preprocess, config, args,
                                                     phase='train', aug=args.aug)
    val_loader_autoL = data_loader_val_autoL

    # -----------------------------------
    # define the loss
    # -----------------------------------
    criterion = pLoss_all_fidelity(hexG=graph_SA_ffso(), dataset='ffpp')

    # load the checkpoint, unless we use the pretrained CLIP weights
    if args.resume and not args.eval:
        max_accuracy = load_checkpoint(config, model_without_ddp, logger)

    # load the dataset
    if not args.aug:
        dataset_train, data_loader_train = set_dataset_singleGPU(args.txt_path_train, preprocess, config, args, phase='train')
    else:
        dataset_train, data_loader_train = set_dataset_singleGPU(args.txt_path_train, preprocess, config, args, phase='train', aug=args.aug)
    dataset_val, data_loader_val = set_dataset_singleGPU(args.txt_path_val, preprocess, config, args, phase='val')

    # define optimizer and scheduler
    optimizer = torch.optim.AdamW(
        params, lr=args.initial_lr,
        weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    log_writer = SummaryWriter(log_dir=config.OUTPUT)
    logger.info("Start training")
    max_accuracy = 0.0
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, args.num_epoch):
        avg_train_loss, avg_train_acc = train_one_epoch_mul_fidelity(config, args, model, joint_texts, val_loader_autoL,
                                                                     meta_optimizer, autol, criterion, data_loader_train,
                                                                     optimizer, epoch, scheduler, log_writer)

        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, args, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        if args.auc:
            acc, auc = validate(config, args, model, joint_texts, criterion, data_loader_val, auc=args.auc)
        else:
            acc = validate(config, args, model, joint_texts, criterion, data_loader_val, auc=args.auc)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc:.4f}%")
        log_writer.add_scalar('Valid/Acc', acc, epoch)
        max_accuracy = max(max_accuracy, acc)
        logger.info(f'Max accuracy: {max_accuracy:.4f}%')

        scheduler.step()

        meta_weight_ls[epoch] = autol.meta_weights.detach().cpu()

        logger.info(get_weight_str(meta_weight_ls[epoch], train_tasks))
        try:
            f.write(get_weight_str(meta_weight_ls[epoch], train_tasks))
            f.write('\n')
        except:
            continue

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info('Best val Acc: {:.2f}'.format(max_accuracy))


def do_batch3_relative_similarity(model, x, joint_texts_prompts):
    batch_size = x.size(0)
    l1_texts = joint_texts_prompts[0][0]
    l2_texts = joint_texts_prompts[1][0]
    l3_texts = joint_texts_prompts[2][0]

    logits_per_image_1, _ = model.forward(x, l1_texts)
    logits_per_image_2, _ = model.forward(x, l2_texts)
    logits_per_image_3, _ = model.forward(x, l3_texts)

    logits_per_image_1 = logits_per_image_1.view(batch_size, -1)
    logits_per_image_2 = logits_per_image_2.view(batch_size, -1)
    logits_per_image_3 = logits_per_image_3.view(batch_size, -1)

    logits_per_image = torch.cat((logits_per_image_1, logits_per_image_2, logits_per_image_3),
                                 dim=1)
    return logits_per_image



def compute_acc_graph(pred, gt):
    targets_gsa = gt
    targets_gsa_binary = targets_gsa

    y_pred_binary = pred[:, 0].data
    y_pred_binary_ls = torch.zeros_like(y_pred_binary) #- 1
    y_pred_binary_ls[y_pred_binary >= 0.5] = 1
    binary_acc = torch.sum(y_pred_binary_ls == targets_gsa_binary).to(torch.float32) / gt.size(0)
    return binary_acc



def train_one_epoch_mul_fidelity(config, args, model, joint_texts, val_loader,
                                 meta_optimizer, autol, criterion, data_loader,
                                 optimizer, epoch, scheduler, log_writer):
    if not args.is_model_train:
        model.eval()
        print('!!!')
    convert_models_to_fp32(model)

    loss_meter = AverageMeter()
    loss_1_meter = AverageMeter()
    loss_2_meter = AverageMeter()
    loss_3_meter = AverageMeter()
    batch_time = AverageMeter()
    acc_meter = AverageMeter()
    num_steps = len(data_loader)

    logger.info(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        logger.info(optimizer.state_dict()['param_groups'][0]['lr'])

    start = time.time()
    end = time.time()

    if args.weight == 'autol':
        val_dataset = iter(val_loader)

    for idx, (samples, targets) in enumerate(data_loader):
        targets = stack_label(targets, dataset='ffpp-so')
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)


        if args.weight == 'autol' and (epoch+1) % args.lambda_epoch==0:
            if epoch == 0 and idx == 0:
                print('yes')
            val_samples, val_label = next(val_dataset)#.next()
            val_label = stack_label(val_label, dataset='ffpp-so')
            val_samples = val_samples.cuda(non_blocking=True)
            val_label = val_label.cuda(non_blocking=True)

            meta_optimizer.zero_grad()

            autol.unrolled_backward(samples, targets, joint_texts,
                                    val_samples, val_label,
                                    scheduler.get_last_lr()[0],
                                    scheduler.get_last_lr()[0], optimizer)

            meta_optimizer.step()

        optimizer.zero_grad()

        with amp.autocast(enabled=True):
            logits_per_image = do_batch3_relative_similarity(model, samples, joint_texts)

            total_loss, pMargin = criterion(logits_per_image, targets, auto_mode=True)
            loss_1 = total_loss[0]
            loss_2 = total_loss[1]
            loss_3 = total_loss[2]
            all_loss = [loss_1, loss_2, loss_3]

            if args.weight == 'autol' and (epoch + 1) % args.lambda_epoch == 0:
                train_loss_tmp = [w * all_loss[i] for i, w in enumerate(autol.meta_weights)]
                loss = sum(train_loss_tmp)
            else:
                loss = total_loss

        loss.backward()
        optimizer.step()

        gt_level_1 = targets[:, 0]
        acc = compute_acc_graph(pMargin, gt_level_1)
        acc_meter.update(acc.item(), targets.size(0))

        loss_meter.update(loss.item(), targets.size(0))
        loss_1_meter.update(loss_1.item(), targets.size(0))
        loss_2_meter.update(loss_2.item(), targets.size(0))
        loss_3_meter.update(loss_3.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_1 {loss_1_meter.val:.4f} ({loss_1_meter.avg:.4f})\t'
                f'loss_2 {loss_2_meter.val:.4f} ({loss_2_meter.avg:.4f})\t'
                f'loss_3 {loss_3_meter.val:.4f} ({loss_3_meter.avg:.4f})\t'
                f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'mem {memory_used:.0f}MB')

            log_writer.add_scalar('Train/total_loss', loss_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))  # *1000
            log_writer.add_scalar('Train/loss_1', loss_1_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/loss_2', loss_2_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/loss_3', loss_3_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/acc', acc_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(config, model, joint_texts, criterion, data_loader, auc=False):
    model.eval()

    batch_time = AverageMeter()
    acc_meter = AverageMeter()

    # for auc
    video_predict = []
    video_label = []

    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        targets = stack_label(targets, dataset='ffpp-so')
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with amp.autocast(enabled=True):
            logits_per_image = do_batch3_relative_similarity(model, samples, joint_texts)

        pMargin = criterion.infer(logits_per_image)
        gt_level_1 = targets[:, 0]
        acc = compute_acc_graph(pMargin, gt_level_1)

        acc_meter.update(acc.item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    if auc:
        auc = AUC(video_label, video_predict) * 100
        logger.info(f' * Acc {acc_meter.avg:.3f}\t'
                    f' * AUC {auc: .3f}')
        return acc_meter.avg, auc
    else:
        logger.info(f' * Acc {acc_meter.avg:.3f}')
        return acc_meter.avg


if __name__ == '__main__':

    args, config = parse_option()

    seed = config.SEED

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    # cudnn.deterministic = True
    # cudnn.benchmark = False

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, args)