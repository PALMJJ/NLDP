import os
import cv2

def edge_extract(train_root):
    label_root = os.path.join(train_root, 'labels')
    edges_root = os.path.join(train_root, 'edges')

    if not os.path.exists(edges_root):
        os.makedirs(edges_root)

    label_list = os.listdir(label_root)
    label_path = []
    for label_name in label_list:
        if not label_name.endswith('.png'):
            assert 'This file %s is not PNG' % (label_name)
        label_path.append(os.path.join(label_root, label_name[:-4] + '.png'))

    index = 0
    for label in label_path:
        label = cv2.imread(label, 0)
        cv2.imwrite(edges_root + '/' + label_list[index], cv2.Canny(label, 30, 100))
        index += 1
    return 0

if __name__ == '__main__':
    train_root = 'dataset/DUTS-TR/'
    edge_extract(train_root)