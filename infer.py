import os
import argparse
import torch
import cv2
from torchvision.transforms import transforms
import time
from vision.network import CRNN
from util.tools import process_img, decode_out


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_file', default='result.txt', type=str, help='file to save results')
parser.add_argument('--width', default=200, type=int, help="input image width")
parser.add_argument('--height', default=32, type=int, help="input image height")
parser.add_argument('--cpu', default=True, help='Use cpu inference')
parser.add_argument('--characters', default="-0123456789", type=str, help="characters")
parser.add_argument('--input_path', default='./test/', type=str, help="image or images dir")
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    net = CRNN(len(args.characters))
    device = torch.device("cpu" if args.cpu else "cuda")
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device(device)))
    net.eval()

    input_path = args.input_path
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    for img_path in image_paths:
        begin = time.time()
        print("recog {}".format(img_path))
        image = cv2.imread(img_path)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img = process_img(image, args.height, args.width, transform)
        net_out = net(img)
        _, preds = net_out.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        lab2str = decode_out(preds, args.characters)
        print(lab2str)
        end = time.time()

    print("Done!!!")