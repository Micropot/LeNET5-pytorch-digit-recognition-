import torch
import numpy as np
import os

def loss_batch(model, loss_func, x, y, opt=None, metric=None):
    pred = model(x)

    loss = loss_func(pred, y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None

    if metric is not None:
        metric_result = metric(pred, y)

    return loss.item(), len(x), metric_result


def evaluate(model, loss_fn, val_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, x, y, metric=metric) for x, y in val_dl]

        losses, nums, metrics = zip(*results)

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total

        avg_metric = None

        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric


def fit(epochs, model, loss_fn, train_dl, val_dl, opt_fn=None, metric=None, scheduler=None, scheduler_on='val_metric'):
    train_losses, val_losses, val_metrics, train_metrics = [], [], [], []
    best_val_loss = 1
    for epoch in range(epochs):

        model.train()
        for x, y in train_dl:
            train_loss, _, train_metric = loss_batch(model, loss_fn, x, y, opt_fn, metric)

        model.eval()
        result = evaluate(model, loss_fn, val_dl, metric)
        val_loss, total, val_metric = result

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        train_metrics.append(train_metric)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.isdir("models"):
                os.mkdir("models")
            print(f"\nBest validation loss: {best_val_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save(model, 'models/best_model.pt'.format(val_loss))

        if metric is None:
            print('Epoch{}/{}, train_loss: {:.4f}, val_loss: {:.4f}'
                  .format(epoch + 1, epochs, train_loss, val_loss))

        else:
            print('Epoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, train_{}: {:.4f}'
                  .format(epoch + 1, epochs, train_loss, val_loss, metric.__name__, val_metric, metric.__name__,
                          train_metric))

        if scheduler is not None:
            if scheduler_on == 'val_metric':
                scheduler.step(val_metrics[-1])



    return train_losses, val_losses, val_metrics, train_metrics


def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)

    return torch.sum(preds == labels).item() / len(preds)