import torch
from torchvision import models, transforms, datasets
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def build_transform(is_train, input_size=224):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(input_size / 0.875)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, args):
    transform = build_transform(is_train)

    if args.data_set == 'cifar10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    # elif args.data_set == 'INAT':
    #     dataset = INatDataset(args.data_path, train=is_train, year=2018,
    #                           category=args.inat_category, transform=transform)
    #     nb_classes = dataset.nb_classes
    # elif args.data_set == 'INAT19':
    #     dataset = INatDataset(args.data_path, train=is_train, year=2019,
    #                           category=args.inat_category, transform=transform)
    #     nb_classes = dataset.nb_classes

    return dataset, nb_classes




# def prepare_cifar10(batch_size):
#     train_transform = build_transform(is_train=True)
#     test_transform = build_transform(is_train=False)
#     # transform = transforms.Compose([
#     #     transforms.Resize((224, 224)),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     # ])

#     trainset = datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=train_transform
#     )
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, num_workers=2
#     )

#     testset = datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=test_transform
#     )
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False, num_workers=2
#     )
#     num_classes = 10
#     return trainloader, testloader, num_classes

