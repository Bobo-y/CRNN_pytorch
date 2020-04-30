import torch.backends.cudnn as cudnn
import math
import time
import datetime
import argparse
import os
from torch.nn import CTCLoss
from vision.network import CRNN
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from util.data_loader import RegDataSet
from util.tools import *


parser = argparse.ArgumentParser("--------Train crnn--------")
parser.add_argument('--weights_save_folder', default='./weights', type=str, help='Dir to save weights')
parser.add_argument('--dataset_root', default='/Users/linyang/PycharmProjects/personal_projects/text_detection/data_util/', help='OCR dataset root')
parser.add_argument('--train_anno', default='annotation_train.txt', help='training imgs annotation txt')
parser.add_argument('--val_anno', default='annotation_val.txt', help='val imgs annotation txt')
parser.add_argument('--lexicon_txt', default='lexicon.txt', help='lexicon txt')
parser.add_argument('--batch_size', default=8, type=int, help="batch size")
parser.add_argument('--width', default=200, type=int, help="input image width")
parser.add_argument('--height', default=32, type=int, help="input image height")
parser.add_argument('--max_epoch', default=50, type=int, help="max training epoch")
parser.add_argument('--initial_lr', default='1e-3', type=float, help="initial learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="gamma for adjust lr")
parser.add_argument('--weight_decay', default=5e-4, type=float, help="weights decay")
parser.add_argument('--characters', default="-0123456789", type=str, help="characters")
parser.add_argument('--num_workers', default=0, type=int, help="numbers of workers")
parser.add_argument('--num_gpu', default=0, type=int, help="gpu number")
parser.add_argument('--pre_train', default=True, type=bool, help="whether use pre-train weights")
args = parser.parse_args()


def val(net, valSet, ctc_loss, max_iter=100):
    net.eval()
    data_loader = DataLoader(valSet, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    val_iter = iter(data_loader)
    i = 0
    loss_avg = 0.0

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        images, labels, target_lengths, input_lengths = next(val_iter)
        i += 1
        preds = net(images)
        cost = ctc_loss(log_probs=preds, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)
        loss_avg += cost
    print("val loss: {}".format(loss_avg / max_iter))
    net.train()


def train(net, optimizer, trainSet, valSet, use_gpu):
    ctc_loss = CTCLoss(blank=0, reduction='mean')
    net.train()
    epoch = 0
    print('Loading Dataset...')

    epoch_size = math.ceil(len(trainSet) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    start_iter = 0

    print("Begin training...")
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            epoch += 1

            batch_iterator = iter(DataLoader(trainSet, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn))
            if epoch % 10 == 0 and epoch > 0:
                if args.num_gpu > 1:
                    torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
                else:
                    torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))

                val(net, valSet, ctc_loss)

        load_t0 = time.time()
        images, labels, target_lengths, input_lengths = next(batch_iterator)
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
            target_lengths = target_lengths.cuda()
            input_lengths = input_lengths.cuda()
        out = net(images)
        optimizer.zero_grad()
        loss = ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths,
                        input_lengths=input_lengths)
        loss.backward()
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f}|| Batchtime: {:.4f} s || ETA: {}'.format
              (epoch, args.max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, loss,
               batch_time, str(datetime.timedelta(seconds=eta))))
    if args.num_gpu > 1:
        torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'Final.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'Final.pth'))
    print('Finished Training')


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    net = CRNN(len(args.characters))

    if args.pre_train:
        pretrained_dict = torch.load(os.path.join(args.weights_save_folder, "Final.pth"))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if args.num_gpu > 1 and use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    elif use_gpu:
        net = net.cuda()

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    trainSet = RegDataSet(dataset_root=args.dataset_root, anno_txt_path=args.train_anno, lexicon_path=args.lexicon_txt,
                          target_size=(args.width, args.height), characters=args.characters, transform=transform)
    valSet = RegDataSet(dataset_root=args.dataset_root, anno_txt_path=args.val_anno, lexicon_path=args.lexicon_txt,
                        target_size=(args.width, args.height), characters=args.characters, transform=transform)
    train(net, optimizer, trainSet, valSet, use_gpu)
