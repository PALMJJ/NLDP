import os
import torch
import argparse
import numpy as np
from scipy import misc
from model.nle import NLE
from crf_process import crf
from dataset import SalientTest
import torch.nn.functional as F

# e -> ECSSD, p -> PASCAL-S, d -> DUT-OMRON, h -> HKU-IS, s -> SOD, m -> MSRA-B, o -> SOC, t -> DUTS-TE
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='e', help='choose test dataset')
opt = parser.parse_args()

# evaluate MAE
def eval_mae(prob, label):
    return np.abs(prob - label).mean()

# get precisions and recalls: threshold --- divided [0, 1] to num values
def eval_pr(prob, label, num):
    prec = np.zeros(num)
    recall = np.zeros(num)
    thlist = np.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (prob >= thlist[i]).astype(float)
        tp = (y_temp * label).sum()
        prec[i] = tp / (y_temp.sum() + 1e-20)
        recall[i] = tp / (label.sum() + 1e-20)
    return prec, recall

def test(num=100, image_size=352):
    model = NLE()
    model.load_state_dict(torch.load('models/DUTS-TR_89.pth'))
    model.cuda()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    dataset_path = ''
    save_path = ''
    if opt.dataset == 'e':
        dataset_path = 'dataset/ECSSD/'
        save_path = 'results/ECSSD/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 'p':
        dataset_path = 'dataset/PASCAL-S/'
        save_path = 'results/PASCAL-S/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 'd':
        dataset_path = 'dataset/DUT-OMRON/'
        save_path = 'results/DUT-OMRON/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 'h':
        dataset_path = 'dataset/HKU-IS/'
        save_path = 'results/HKU-IS/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 's':
        dataset_path = 'dataset/SOD/'
        save_path = 'results/SOD/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 'm':
        dataset_path = 'dataset/MSRA-B/'
        save_path = 'results/MSRA-B/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 'o':
        dataset_path = 'dataset/SOC/'
        save_path = 'results/SOC/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif opt.dataset == 't':
        dataset_path = 'dataset/DUTS-TE/'
        save_path = 'results/DUTS-TE/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    image_root = dataset_path + 'images/'
    label_root = dataset_path + 'labels/'
    test_dataset = SalientTest(image_root, label_root, image_size)

    beta = np.sqrt(0.3)
    avg_mae = 0.0
    avg_prec = np.zeros(num)
    avg_recall = np.zeros(num)
    for i in range(test_dataset.__len__()):
        image, label, name = test_dataset[i]
        image = image.to(device)
        label = label.to(device)
        _, _, prob, _ = model(image)
        prob = F.interpolate(prob, (label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        prob = prob.sigmoid().data.cpu().detach().numpy()
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        image = F.interpolate(image, (label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        image = image.data.cpu().detach().numpy().squeeze().reshape(label.shape[1], label.shape[2], -1).astype('uint8')
        prob = crf(image, prob, to_tensor=False)
        label = label.cpu().detach().numpy().squeeze()
        avg_mae += eval_mae(prob, label).item()
        prec, recall = eval_pr(prob, label, num)
        avg_prec += prec
        avg_recall += recall
        prob = prob.reshape((prob.shape[1], prob.shape[2]))
        misc.imsave(save_path + name, prob)
    avg_mae = avg_mae / len(test_dataset)
    avg_prec = avg_prec / len(test_dataset)
    avg_recall = avg_recall / len(test_dataset)
    F_beta = (1 + beta ** 2) * avg_prec * avg_recall / (beta ** 2 * avg_prec + avg_recall)
    F_beta[F_beta != F_beta] = 0
    avgF = F_beta.mean()
    maxF = F_beta.max()
    print(avg_prec)
    print(avg_recall)

    return avg_mae, avg_prec, avg_recall, avgF, maxF

dataset_name = ''
if opt.dataset == 'e':
    dataset_name = 'ECSSD'
elif opt.dataset == 'p':
    dataset_name = 'PASCAL-S'
elif opt.dataset == 'd':
    dataset_name = 'DUT-OMRON'
elif opt.dataset == 'h':
    dataset_name = 'HKU-IS'
elif opt.dataset == 's':
    dataset_name = 'SOD'
elif opt.dataset == 'm':
    dataset_name = 'MSRA-B'
elif opt.dataset == 'o':
    dataset_name = 'SOC'
elif opt.dataset == 't':
    dataset_name = 'DUTS-TE'

print('Starting')
avg_mae, avg_prec, avg_recall, avgF, maxF = test()
print(dataset_name + ' Average mean absolute error:', avg_mae)
print(dataset_name + ' Average F-measure:', avgF)
print(dataset_name + ' Max F-measure:', maxF)

logs_root = 'logs/'
if not os.path.exists(logs_root):
    os.makedirs(logs_root)
with open(logs_root + 'log.txt', 'a') as f:
    f.writelines(dataset_name + ' Average mean absolute error:' + str(avg_mae) + '\n')
    f.writelines(dataset_name + ' Average F-measure:' + str(avgF) + '\n')
    f.writelines(dataset_name + ' Max F-measure:' + str(maxF) + '\n')