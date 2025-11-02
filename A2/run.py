#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import os
import math
import time
import argparse
import torch
from torch import nn, optim
from tqdm import tqdm
from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description='Train neural dependency parser in pytorch')
parser.add_argument('-d', '--debug', action='store_true', help='whether to enter debug mode')
args = parser.parse_args()

def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    best_dev_UAS = 0

    # 1) Optimizer and Loss
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} out of {n_epochs}")
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    parser.model.train()
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            # 2) Forward + Loss + Backprop + Step
            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()

            prog.update(1)
            loss_meter.update(loss.item())

    print(f"Average Train Loss: {loss_meter.avg}")
    print("Evaluating on dev set")
    parser.model.eval()
    dev_UAS, _ = parser.parse(dev_data)
    print(f"- dev UAS: {dev_UAS * 100.0:.2f}")
    return dev_UAS


if __name__ == "__main__":
    debug = args.debug
    assert (torch.__version__.split(".") >= ["1", "0", "0"]), "Please install torch version >= 1.0.0"

    print("=" * 80)
    print("INITIALIZING")
    print("=" * 80)
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print(f"took {time.time() - start:.2f} seconds\n")

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    output_dir = f"results/{datetime.now():%Y%m%d_%H%M%S}/"
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print("=" * 80)
        print("TESTING")
        print("=" * 80)
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set")
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print(f"- test UAS: {UAS * 100.0:.2f}")
        print("Done!")
