from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import clip
import random, json, os, argparse, pdb
import numpy as np
from timm.utils import AverageMeter
from logger import create_logger
from sklearn.metrics import roc_auc_score as AUC
from utils import load_checkpoint
from set_dataset_test import set_dataset_singleGPU_CDF, set_dataset_singleGPU_FSh, set_dataset_singleGPU_Deeper, \
    set_dataset_singleGPU_DFDC, MyDataset_FFSC, MyDataset_Diffusion
from config_test import get_config
from SO_Loss import pLoss_all_fidelity
from SO_Graph import graph_SA_ffso, graph_SO_FFSC
from torch.cuda import amp
import torchvision
from torchvision.transforms import InterpolationMode
from PIL import Image


def parse_option():
    parser = argparse.ArgumentParser('SJEDD', add_help=False)
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
    parser.add_argument('--test_log', action='store_true')

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--name', default='SJEDD Cross-dataset Test', type=str)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')

    parser.add_argument('--resume', help='resume from checkpoint')

    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--test_log', type=str, default='True')
    # for CDF, FSh
    parser.add_argument('--dataset', type=str, default='CDF', help='FSh, Deeper')
    parser.add_argument('--datapath', type=str, default='/data0/mian2/celeb-df/dataset/', help='/data0/mian2/FaceShifter/, /data0/mian2/DeeperForensics/')
    parser.add_argument('--n_frames', type=int, default=32)

    parser.add_argument('--train_dataset', type=str, default="ffpp", help='ffsc')
    parser.add_argument('--VM', type=str, default="ViT-B/32", help='RN50, RN101, ViT-B/16, ViT-L/14')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config, args):
    # ---------------------------------
    # define the templates
    # ---------------------------------
    l1_level = ['fake']

    if args.train_dataset == 'ffpp':
        l2_level = ['expression', 'identity', 'physical inconsistency']
        l3_level = ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin']

    elif args.train_dataset == 'ffsc':
        l2_level = ['age', 'expression', 'gender', 'identity', 'pose']
        l3_level = ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin']

    else:
        raise NotImplementedError

    l1_texts = torch.cat([clip.tokenize(f"A photo of a {l1} face") for l1 in l1_level]).to('cuda').unsqueeze(0)

    l2_texts = torch.cat(
        [clip.tokenize(f"A photo of a face with the global attribute of {l2} altered") for l2 in l2_level]).to(
        'cuda').unsqueeze(0)

    l3_texts = torch.cat(
        [clip.tokenize(f"A photo of a face with the local attribute of {l3} altered") for l3 in l3_level]).to(
        'cuda').unsqueeze(0)

    joint_texts = [l1_texts, l2_texts, l3_texts]

    # -----------------------------------
    # define the visual model
    # -----------------------------------
    mode_name = args.VM # default setting is "ViT-B/32"
    model, preprocess, _, __ = clip.load(mode_name, device='cuda', jit=False)
    model.cuda()
    model_without_ddp = model

    # -----------------------------------
    # define inference loss
    # -----------------------------------
    if args.train_dataset == 'ffpp':
        criterion = pLoss_all_fidelity(hexG=graph_SA_ffso(), dataset='ffpp')
    elif args.train_dataset == 'ffsc':
        criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC(), dataset='ffsc')
    # load the checkpoint, unless we use the pretrained CLIP weights

    # for test
    if args.dataset == 'FFSC':
        txt_ffsc_path = args.datapath  # "/data0/mian2/vision-language/FFSC-test.txt"
        dataset_val = MyDataset_FFSC(txt_ffsc_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False,
                                                  pin_memory=True, num_workers=2)
        if args.resume:
            max_accuracy = load_checkpoint(config, model_without_ddp, logger)
        _, auc_ffsc = test_imgs(config, args, model, joint_texts, test_loader, criterion, auc=True)


    if args.dataset == 'diffusion':
        try:
            BICUBIC = InterpolationMode.BICUBIC
        except ImportError:
            BICUBIC = Image.BICUBIC

        def custom_collate_fn(batch):
            images, labels = zip(*batch)
            processed_images = []
            for img in images:
                # 确保每张图片都是 (3, 224, 224)
                if img.size(1) != 224 or img.size(2) != 224:
                    img = torchvision.transforms.Resize((224, 224), interpolation=BICUBIC)(img)
                processed_images.append(img)
            images = torch.stack(processed_images)
            labels = torch.tensor(labels)
            return images, labels

        txt_diffusion_path = args.datapath
        dataset_val = MyDataset_Diffusion(txt_diffusion_path, transform=preprocess)
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False,
                                                  pin_memory=True, num_workers=2,
                                                  collate_fn=custom_collate_fn)
        if args.resume:
            max_accuracy = load_checkpoint(config, model_without_ddp, logger)
        acc, auc = test_imgs(config, args, model, joint_texts, test_loader, criterion, auc=True)


    elif args.dataset == 'CDF':
        dataset_val, data_loader_val = set_dataset_singleGPU_CDF(config, preprocess, args.datapath, args.n_frames)
        # load the checkpoint
        if args.resume:
            max_accuracy = load_checkpoint(config, model_without_ddp, logger)
        if args.auc:
            acc, auc = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)
        else:
            acc = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)

    elif args.dataset == 'FSh':
        dataset_val, data_loader_val = set_dataset_singleGPU_FSh(config, preprocess, '/data0/mian2/FaceShifter/',
                                                                 args.n_frames)
        # load the checkpoint
        if args.resume:
            max_accuracy = load_checkpoint(config, model_without_ddp, logger)
        _, auc_fsh = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)


    elif args.dataset == 'Deeper':
        dataset_val, data_loader_val = set_dataset_singleGPU_Deeper(config, preprocess, '/data0/mian2/DeeperForensics/',
                                                                    args.n_frames)
        if args.resume:
            max_accuracy = load_checkpoint(config, model_without_ddp, logger)
        if args.auc:
            acc, auc = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)
        else:
            acc = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)

    elif args.dataset == 'DFDC':
        dataset_val, data_loader_val = set_dataset_singleGPU_DFDC(config, preprocess, args.n_frames)
        # load the checkpoint
        if args.resume:
            # load the checkpoint
            max_accuracy = load_checkpoint(config, model_without_ddp, logger)
        if args.auc:
            acc, auc = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)
        else:
            acc = test(config, args, model, joint_texts, data_loader_val, criterion, auc=args.auc)





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



@torch.no_grad()
def test(config, args, model, joint_texts, data_loader, criterion, auc=False):
    fo = open(config.TEST_LOG, 'a')
    model.eval()

    acc_meter = AverageMeter()
    # for auc
    video_predict = []
    video_label = []

    for idx, (samples, targets) in enumerate(tqdm(data_loader)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with amp.autocast(enabled=True):
            logits_per_image = do_batch3_relative_similarity(model, samples.squeeze(0), joint_texts)
        probs = criterion.infer(logits_per_image)[:, 0]

        probs_auc = probs
        frames_predict = probs_auc.detach().cpu().numpy().tolist()
        video_predict.append(np.mean(frames_predict))
        video_label.append(targets.detach().cpu().numpy())

    if auc:
        auc = AUC(video_label, video_predict) * 100
        logger.info(f' * AUC {auc: .3f}')
        if args.test_log:
            fo.write(f' * AUC {auc:.3f}')
            fo.write('\n')
        return acc_meter.avg, auc
    else:
        logger.info(f' * Acc {acc_meter.avg:.3f}')
        return acc_meter.avg

@torch.no_grad()
def test_imgs(config, args, model, joint_texts, data_loader, criterion, auc=False):
    fo = open(config.TEST_LOG, 'a')
    model.eval()

    acc_meter = AverageMeter()
    # for auc
    video_predict = []
    video_label = []

    for idx, (samples, targets) in enumerate(tqdm(data_loader)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with amp.autocast(enabled=True):
            logits_per_image = do_batch3_relative_similarity(model, samples, joint_texts)
        probs = criterion.infer(logits_per_image)[:, 0]
        probs_auc = probs

        frames_predict = probs_auc.detach().cpu().numpy().tolist()
        video_predict.extend(frames_predict)
        video_label.extend(targets.detach().cpu().numpy().tolist())

    if auc:
        auc = AUC(video_label, video_predict) * 100
        logger.info(f' * AUC {auc: .3f}')
        if args.test_log:
            fo.write(f' * AUC {auc:.3f}')
            fo.write('\n')
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