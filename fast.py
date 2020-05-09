from transformer import TransformerNet
from utils import load_image, save_image, gram_matrix, normalize_batch
from vgg import Vgg16

from PIL import Image
from collections import namedtuple
from torchvision import models, datasets, transforms
import os
import time
import re
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


def train(style_image, dataset_path):
    print('Training function started...')
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 256
    style_weight = 1e10
    content_weight = 1e5
    lr = 1e-3
    batch_size = 3
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(dataset_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet().to(device)

    optimizer = Adam(transformer.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16().to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = load_image(style_image)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]
    epochs = 2
    print('Starting epochs...')
    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])

            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            log_interval = 2000
            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

    # save model
    transformer.eval().cpu()
    save_model_path = 'models/outpost.pth'
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(content_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_name = 'results/'+'output_'+content_image
    content_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load('models/outpost.pth')
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        output = style_model(content_image).cpu()

    save_image(output_name, output[0])


def main(training=False):
    if training:
        train('styles/outpost.jpg', 'dataset')
    stylize_list = ['profilowe.jpg', 'twarz.jpg', 'mojatwarz.jpg', 'zamosc.jpg', 'budynek.jpg']
    for style_picture in stylize_list:
        stylize('contents/' + style_picture)


if __name__ == '__main__':
    main(training=True)
