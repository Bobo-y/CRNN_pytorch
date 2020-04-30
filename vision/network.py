import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs['D'])

    def forward(self, x):
        x = self.features(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    i = 0
    for v in cfg:
        if v == 'M':
            if i not in [9, 13]:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))]

        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
}


class BidirectionalLSTM(nn.Module):

    def __init__(self, inp, nHidden, oup):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(inp, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, oup)

    def forward(self, x):
        out, _ = self.rnn(x)
        T, b, h = out.size()
        t_rec = out.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, characters_classes, hidden=256, pretrain=True):
        super(CRNN, self).__init__()
        self.characters_class = characters_classes
        self.body = VGG()
        # 将VGG stage5-1 卷积单独拿出来, 改了卷积核无法加载预训练参数
        self.stage5 = nn.Conv2d(512, 512, kernel_size=(3, 2), padding=(1, 0))
        self.hidden = hidden
        self.rnn = nn.Sequential(BidirectionalLSTM(512, self.hidden, self.hidden),
                                 BidirectionalLSTM(self.hidden, self.hidden, self.characters_class))

        self.pretrain = pretrain
        if self.pretrain:
            import torchvision.models.vgg as vgg
            pre_net = vgg.vgg16(pretrained=True)
            pretrained_dict = pre_net.state_dict()
            model_dict = self.body.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.body.load_state_dict(model_dict)

            for param in self.body.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.body(x)
        x = self.stage5(x)
        x = x.squeeze(3)
        x = x.permute(2, 0, 1).contiguous()
        x = self.rnn(x)
        x = F.log_softmax(x, dim=2)
        return x
