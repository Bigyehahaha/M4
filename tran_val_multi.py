import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
def train(args, model, optimizer, data_loader, device, epoch):
    model.train()
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    train_loss = 0.0
    train_loss_single = []
    for _ in range((args.num_classes)):
        train_loss_single.append(0.0)

    loss_functions = nn.BCELoss(reduction='mean')

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))
        labels = labels.to(device)

        loss = []
        for i in range(len(pred)): 

            task_loss = loss_functions(pred[i].view(-1, 1), labels[:, i].view(-1, 1).to(torch.float))

            loss.append(task_loss)
            train_loss_single[i] += task_loss

        optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(data_loader)
    for i in range(len(train_loss_single)):
        train_loss_single[i] /= len(data_loader)
    print('Training loss:{}'.format(train_loss))

    return train_loss


@torch.no_grad()
def evaluate(args, model, data_loader, device):
    model.eval()
    loss_functions = nn.BCELoss(reduction='mean')
    total_num = len(data_loader.dataset)

    # Initialize lists for predictions and targets based on args.num_classes
    preds = [[] for _ in range(args.num_classes)]
    targets = [[] for _ in range(args.num_classes)]
    binary_preds = [[] for _ in range(args.num_classes)]
    data_loader = tqdm(data_loader, file=sys.stdout)
    eval_loss_all = 0.0
    eval_loss_single = [0.0 for _ in range(args.num_classes)]
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        labels = labels.to(device)
        
        eval_loss = []
        for i in range(args.num_classes):
            y = labels[:, i]
            y_pred = pred[i]
            
            task_loss = loss_functions(y_pred.view(-1, 1), y.view(-1, 1).to(torch.float))
            eval_loss.append(task_loss)
            eval_loss_single[i] += task_loss.item()
            
            # Collect predictions and targets
            preds[i].append(y_pred.squeeze(0).cpu().numpy())
            binary_preds[i].append((y_pred.squeeze(0).cpu().numpy()>=0.5).astype(int))
            targets[i].append(y.cpu().numpy())
        
        eval_loss_all += sum(eval_loss)
    
    eval_loss_all /= len(data_loader)
    for i in range(args.num_classes):
        eval_loss_single[i] /= len(data_loader)
    
    # Calculate AUC for each class
    AUC = []
    for i in range(args.num_classes):
        auc = roc_auc_score(targets[i], preds[i])
        AUC.append(auc)
    # Calculate F1, ACC, Sensitivity and Specificity for each class
    F1 = []
    ACC = []
    Sen = []
    Spe = []
    for i in range(args.num_classes):
        tn, fp, fn, tp = confusion_matrix(np.array(targets[i]), np.where(np.array(preds[i]).reshape(-1,1)>0.5,1,0)).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        F1.append(f1)
        ACC.append(acc)
        Sen.append(sensitivity)
        Spe.append(specificity)
    return AUC, F1, ACC, Sen, Spe

def train_single(args, model, optimizer, data_loader, device):
    model.train()
    loss_function = torch.nn.BCELoss(reduction='mean')
    optimizer.zero_grad()

    total_num = len(data_loader.dataset)
    data_loader = tqdm(data_loader, file=sys.stdout)
    train_loss = 0

    for step, data in enumerate(data_loader):
        images, labels= data
        labels = labels.unsqueeze(0)
        pred = model(images.to(device))
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        
        if args.num_classes == 1:
            loss = loss_function(pred, labels.squeeze(0).to(device).float())
        loss_item = loss.item()
        optimizer.zero_grad()
        train_loss += loss_item
        loss.backward()

        pred = torch.max(pred, dim=1)[1]

        optimizer.step()
        

    train_loss /= len(data_loader)

    print('Training loss:{}'.format(train_loss))

    return train_loss


@torch.no_grad()
def evaluate_single(args, model, data_loader, device):
    model.eval()

    loss_function = torch.nn.BCELoss(reduction='mean')
    mean_loss_val = torch.zeros(1).to(device)
    total_num = len(data_loader.dataset)
    auc_score = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)

    auc_sum = 0

    targets = []
    preds = []

    val_loss = 0
    thresholds = []
    thresholds_optimal = []
    aucs = []
    y_pred = []
    y_tru = []


    for step, data in enumerate(data_loader):
        images, labels= data
        pred = model(images.to(device))
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        targets.append(labels.cpu().numpy())
        preds.extend(pred.cpu().numpy())

        if args.num_classes == 1:
            loss_val = loss_function(pred, labels.to(device).float())

        val_loss += loss_val.item()

        pred = torch.max(pred, dim=1)[1]
        y_tru.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

    val_loss /= len(data_loader)

    targets = np.array(targets).squeeze(-1)
    preds = np.array(preds)

    auc = roc_auc_score(targets, preds)

    tn, fp, fn, tp = confusion_matrix(np.array(targets), np.where(np.array(preds).reshape(-1,1)>0.5,1,0)).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    return auc, f1, acc, sensitivity, specificity