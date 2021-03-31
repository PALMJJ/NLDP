import os
import torch
import argparse
from model.nle import NLE
from datetime import datetime
from dataset import get_loader
from torch.autograd import Variable
from utils import clip_gradient, adjust_lr

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--image_size', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NLE().to(device)
parameters = model.parameters()
optimizer = torch.optim.Adam(parameters, opt.lr)

image_root = 'dataset/DUTS-TR/images/'
label_root = 'dataset/DUTS-TR/labels/'
train_loader = get_loader(image_root, label_root, image_size=opt.image_size, batch_size=opt.batch_size)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()

def train(train_loader, model, optimizer, epoch, dataset_name):
    model.train()
    for i, pack in enumerate(train_loader):
        optimizer.zero_grad()

        images, labels = pack
        images = Variable(images)
        labels = Variable(labels)
        images = images.to(device)
        labels = labels.to(device)

        init_prob1, init_prob2, final_prob1, final_prob2 = model(images)
        init_loss1 = CE(init_prob1, labels)
        init_loss2 = CE(init_prob2, labels)
        final_loss1 = CE(final_prob1, labels)
        final_loss2 = CE(final_prob2, labels)

        loss = init_loss1 + init_loss2 + final_loss1 + final_loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 400 == 0 or i == total_step:
            info = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data)
            print(info)
            logs_root = 'logs/'
            if not os.path.exists(logs_root):
                os.makedirs(logs_root)
            with open(logs_root + 'log.txt', 'a') as f:
                f.writelines(info + '\n')

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), save_path + dataset_name + '_' + str(epoch) + '.pth')

for epoch in range(opt.epoch):
    adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch, 'DUTS-TR')