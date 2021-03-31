import os
from PIL import Image
from torch.utils import data
from torchvision import transforms

class SalientTrain(data.Dataset):
    def __init__(self, image_root, label_root, image_size):
        super(SalientTrain, self).__init__()
        self.image_size = image_size
        self.image_list = list(map(lambda x: os.path.join(image_root, x), os.listdir(image_root)))
        self.label_list = list(map(lambda x: os.path.join(label_root, x), os.listdir(label_root)))
        self.image_list = sorted(self.image_list)
        self.label_list = sorted(self.label_list)
        self.filter_files()
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def filter_files(self):
        assert len(self.image_list) == len(self.label_list)
        image_list = []
        label_list = []
        for image_path, label_path in zip(self.image_list, self.label_list):
            image = Image.open(image_path)
            label = Image.open(label_path)
            if image.size == label.size:
                image_list.append(image_path)
                label_list.append(label_path)
        self.image_list = image_list
        self.label_list = label_list

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('L')
        image = self.image_transform(image)
        label = self.label_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_list)

def get_loader(image_root, label_root, image_size, batch_size, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SalientTrain(image_root, label_root, image_size)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader

class SalientTest(data.Dataset):
    def __init__(self, image_root, label_root, image_size):
        super(SalientTest, self).__init__()
        self.image_size = image_size
        self.image_list = list(map(lambda x: os.path.join(image_root, x), os.listdir(image_root)))
        self.label_list = list(map(lambda x: os.path.join(label_root, x), os.listdir(label_root)))
        self.image_list = sorted(self.image_list)
        self.label_list = sorted(self.label_list)
        self.image_tensor_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_transform = transforms.ToTensor()
        self.label_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('L')
        image_tensor = self.image_tensor_transform(image).unsqueeze(0)
        image = self.image_transform(image).unsqueeze(0)
        label = self.label_transform(label)
        name = self.image_list[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image_tensor, image, label, name

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    trainDataset = SalientTrain("dataset/DUTS-TR/images", "dataset/DUTS-TR/labels", 352)
    testDataset = SalientTest("dataset/DUTS-TR/images", "dataset/DUTS-TR/labels", 352)
    image, label = trainDataset[1]
    print(image.shape)
    print(label.shape)
    image, label, name = testDataset[1]
    print(image.shape)
    print(label.shape)
    """
    torch.Size([3, 352, 352])
    torch.Size([1, 352, 352])
    torch.Size([1, 3, 352, 352])
    torch.Size([1, 300, 400])
    """