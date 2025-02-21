
import argparse
import csv
import pdb
import os
import math

from train_val_multi import *
from Dataset import *

from sklearn.model_selection import train_test_split
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import numpy as np
#multi_modal
from models.MMoE import MMoE
from models.M4 import M4
#single_modal
from models.AMIL import PorpoiseAMIL
def main(args):
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_file)
    img_path = args.features_path
    df['slide_path'] = img_path + df['slide_id'] + '.pt'

    label_col = df.columns.tolist()

    label_col.remove('case_id')
    label_col.remove('slide_id')
    label_col.remove('slide_path')

    label = {header: df[header].tolist() for header in label_col}

    label_map = {}
    for label, values in label.items():
        unique_values = sorted(set(values))
        label_map[label] = {value: idx for idx, value in enumerate(unique_values)}

    for label, mapping in label_map.items():
        df[label] = df[label].apply(lambda x: mapping[x])

    print(label_map)

    save_files = os.path.join(args.save_path)

    if not os.path.exists(save_files):
        parent_folder = os.path.dirname(save_files)
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder) 

    X = df["slide_path"].values
    y = df[label_col].values

    # Split the dataset into training set, validation set, and testing set (divided by patients)
    unique_patients = df["case_id"].unique()
    train_val_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    best_auc_fold = np.zeros(args.num_folds)
    f1_fold = np.zeros(args.num_folds)
    acc_fold = np.zeros(args.num_folds)
    sen_fold = np.zeros(args.num_folds)
    spe_fold = np.zeros(args.num_folds)
    for i in range(5):#5-fold cross-validation
        if os.path.exists(f'{save_files}/s_{i}_checkpoint.pt') is True:
            print("This epoch exists!")
            continue
        SEED = args.seed
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('fold {}'.format(i))
        train_patient, val_patient = train_test_split(train_val_patients, test_size=0.2, random_state=i)
        train_indices = df[df["case_id"].isin(train_patient)].index
        val_indices = df[df["case_id"].isin(val_patient)].index

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        train_indices_df = pd.DataFrame({"train_indices": list(X_train)})
        val_indices_df = pd.DataFrame({"val_indices": list(X_val)})
        train_label_indices_df = pd.DataFrame({"train_indices_label": list(y_train)})
        val_label_indices_df = pd.DataFrame({"val_indices_label": list(y_val)})
        if os.path.isdir(save_files):
           pass
        else:
            os.makedirs(save_files) 
        train_indices_df.to_csv(f"{save_files}/train_indices_fold_{i}.csv", index=False)
        val_indices_df.to_csv(f"{save_files}/val_indices_fold_{i}.csv", index=False)

        print("Training set sample number:", len(X_train))
        print("Validation set sample number:", len(X_val))

        if os.path.exists(f'{save_files}/weights_{i}') is False:
            os.mkdir(f'{save_files}/weights_{i}')

        SWriter = SummaryWriter(log_dir=f'{save_files}/fold_{i}')

        slide_train, label_train = train_indices_df['train_indices'].tolist(), train_label_indices_df[
            'train_indices_label'].tolist()
        slide_val, label_val = val_indices_df['val_indices'].tolist(), val_label_indices_df[
            'val_indices_label'].tolist()
        
        train_data = Datasets(args=args, fts_path=slide_train, fts_label=label_train)
        val_data = Datasets(args=args, fts_path=slide_val, fts_label=label_val)

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate_MIL,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate_MIL,
            pin_memory=True,
            drop_last=True
        )
        if args.multi is True:
            if args.model == 'MMoE':
                model = MMoE(feat_input=args.feat_input, experts_out=args.experts_out, towers_out=1, towers_hidden=32, tasks=args.num_classes, num_expert=args.num_expert).to(
                    device)
            elif args.model == 'M4':
                model = M4(feat_input=args.feat_input, experts_out=args.experts_out, towers_out=1, towers_hidden=32, tasks=args.num_classes, num_expert=args.num_expert).to(
                    device)
            else:
                pass
        else:
            if args.model == 'AMIL':
                model = PorpoiseAMIL(feat_input=args.feat_input, n_classes=args.num_classes).to(device)
            else:
                pass
            
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(pg, lr=args.lr, weight_decay=4E-5)

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        best_auc = 0
        for epoch in range(args.epochs):
            if args.multi is True:#multi-task
                mean_loss = train(args, model=model,
                                  optimizer=optimizer,
                                  data_loader=train_loader,
                                  device=device,
                                  epoch=epoch
                                  )

                scheduler.step()

                AUC,F1,ACC,Sen,Spe = evaluate(args, model=model,data_loader=val_loader,device=device)

                mean_auc = sum(AUC)/len(AUC)
                mean_f1 = sum(F1)/len(F1)
                mean_acc = sum(ACC)/len(ACC)
                mean_sen = sum(Sen)/len(Sen)
                mean_spe = sum(Spe)/len(Spe)
                print(f'Epoch{epoch},Average AUC:{mean_auc}, Average F1-Score:{mean_f1}, Average ACC:{mean_acc}, Average Sensitivity:{mean_sen}, Average Specificity:{mean_spe}.')
                
                if best_auc < mean_auc:
                    best_auc = mean_auc
                    print('New best_auc_mean: {}'.format(best_auc))
                    best_auc_fold[i] = best_auc
                    f1_fold[i] = mean_f1
                    acc_fold[i] = mean_acc
                    torch.save(model.state_dict(), f'{save_files}/s_{i}_checkpoint.pt')

                # print("[epoch {}] auc: {}".format(epoch, round(auc, 3)))
                tags = ["train_loss", "auc_mean", "f1-score", "acc", "sen", "spe"]
                SWriter.add_scalar(tags[0], mean_loss, epoch)
                SWriter.add_scalar(tags[1], mean_auc, epoch)
                SWriter.add_scalar(tags[2], mean_f1, epoch)
                SWriter.add_scalar(tags[3], mean_acc, epoch)
                SWriter.add_scalar(tags[4], mean_sen, epoch)
                SWriter.add_scalar(tags[5], mean_spe, epoch)

            else:#single-task
                mean_loss = train_single(args, model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device)
                scheduler.step()
                auc, f1, acc, sen, spe = evaluate_single(args, model=model, data_loader=val_loader, device=device)

                print(f'Epoch{epoch}, AUC{auc}, F1-Score{f1}, ACC{acc}, Sensitivity{sen}, Specificity{spe}.')
                if best_auc < auc:
                    best_auc = auc
                    print('best_auc_mean: {}'.format(best_auc))
                    best_auc_fold[i] = best_auc
                    f1_fold[i] = f1
                    acc_fold[i] = acc
                    sen_fold[i] = sen
                    spe_fold[i] = spe
                    torch.save(model.state_dict(), f'{save_files}/s_{i}_checkpoint.pt')

                tags = ["train_loss", "auc", "f1", "acc","sen","spe"]
                SWriter.add_scalar(tags[0], mean_loss, epoch)
                SWriter.add_scalar(tags[1], auc, epoch)
                SWriter.add_scalar(tags[2], f1, epoch)
                SWriter.add_scalar(tags[3], acc, epoch)
                SWriter.add_scalar(tags[4], sen, epoch)
                SWriter.add_scalar(tags[5], spe, epoch)
            
    save_file = open(f'{save_files}/auc_f1_acc.csv', 'w')
    csv_writer = csv.writer(save_file)
    csv_writer.writerow(['fold', 'auc', 'f1-score', 'acc'])
    for i in range(args.num_folds):
        csv_writer.writerow([i, best_auc_fold[i], f1_fold[i], acc_fold[i]])
    save_file.close()


    test_auc_mean = np.zeros(args.num_folds)
    test_f1_score_mean = np.zeros(args.num_folds)
    test_acc_mean = np.zeros(args.num_folds)
    test_sen = np.zeros(args.num_folds)
    test_spe = np.zeros(args.num_folds)
    # Initialize lists based on args.num_classes
    auc_lists = [[] for _ in range(args.num_classes)]
    f1_lists = [[] for _ in range(args.num_classes)]
    acc_lists = [[] for _ in range(args.num_classes)]
    sen_lists = [[] for _ in range(args.num_classes)]
    spe_lists = [[] for _ in range(args.num_classes)]
    for i in range(5):
        # test
        print(f'Test fold {i}:')
        test_indices = df[df["case_id"].isin(test_patients)].index
        X_test, y_test = X[test_indices], y[test_indices]

        test_indices_df = pd.DataFrame({"test_indices": list(X_test)})
        test_label_indices_df = pd.DataFrame({"test_indices_label": list(y_test)})
        test_indices_df.to_csv(f"{save_files}/test_indices.csv", index=False)

        print("Test set sample number:", len(X_test))

        slide_test, label_test = test_indices_df['test_indices'].tolist(), test_label_indices_df[
            'test_indices_label'].tolist()

        if args.multi is True:
            test_data = Dataset_multi(args=args, fts_path=slide_test, fts_label=label_test)
        else:
            test_data = Dataset_single(args=args, fts_path=slide_test, fts_label=label_test)

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_MIL,
            pin_memory=True
        )
        if args.multi is True:
            if args.model == 'MMoE':
                model = MMoE(feat_input=args.feat_input, experts_out=args.experts_out, towers_out=1, towers_hidden=32, tasks=args.num_classes, num_expert=args.num_expert).to(
                    device)
            elif args.model == 'M4':
                model = M4(feat_input=args.feat_input, experts_out=args.experts_out, towers_out=1, towers_hidden=32, tasks=args.num_classes, num_expert=args.num_expert).to(
                    device)
            else:
                pass
        else:
            if args.model == 'AMIL':
                model = PorpoiseAMIL(feat_input=args.feat_input, n_classes=args.num_classes).to(device)
            else:
                pass
        if os.path.exists(f'{save_files}/s_{i}_checkpoint.pt') is True:
            model_weight_path = os.path.join((f'{save_files}/s_{i}_checkpoint.pt'))
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
        else:
            print('No model weight found!')

        if args.multi is True:
            AUC, F1_score, ACC, Sen, Spe = evaluate(args, model=model, data_loader=test_loader, device=device)
            for idx, auc in enumerate(AUC):
                auc_lists[idx].append(auc)
                f1_lists[idx].append(F1_score[idx])
                acc_lists[idx].append(ACC[idx])
                sen_lists[idx].append(Sen[idx])
                spe_lists[idx].append(Spe[idx])
            auc = sum(AUC)/len(AUC)
            f1_score = sum(F1_score)/len(F1_score)
            acc = sum(ACC)/len(ACC)
            sen = sum(Sen)/len(Sen)
            spe = sum(Spe)/len(Spe)
            #Print auc, f1_score, acc, sen, spe for each class
            print(f'Mean auc: ' + ', '.join(f'auc_{i+1}: {sum(auc_lists[i]) / 5}' for i in range(args.num_classes)))
            print(f'Mean f1: ' + ', '.join(f'f1_{i+1}: {sum(f1_lists[i]) / 5}' for i in range(args.num_classes)))
            print(f'Mean acc: ' + ', '.join(f'acc_{i+1}: {sum(acc_lists[i]) / 5}' for i in range(args.num_classes)))
            print(f'Mean sen: ' + ', '.join(f'sen_{i+1}: {sum(sen_lists[i]) / 5}' for i in range(args.num_classes)))
            print(f'Mean spe: ' + ', '.join(f'spe_{i+1}: {sum(spe_lists[i]) / 5}' for i in range(args.num_classes)))
    
        elif args.multi is False:
              auc, f1_score, acc, sen, spe = evaluate_single(args, model=model, data_loader=test_loader, device=device)
        test_auc_mean[i] = auc
        test_f1_score_mean[i] = f1_score
        test_acc_mean[i] = acc
        test_sen[i] = sen
        test_spe[i] = spe

    save_file_test = open(f'{save_files}/test_auc_f1_acc.csv', 'w')
    csv_writer = csv.writer(save_file_test)
    csv_writer.writerow(['fold', 'auc', 'f1_score', 'acc'])
    #save auc, f1_score, acc, sen, spe for each fold
    for i in range(args.num_folds):
        csv_writer.writerow([i, test_auc_mean[i],test_f1_score_mean[i], test_acc_mean[i], test_sen[i], test_spe[i]])
    #save average results
    csv_writer.writerow(['average_AUC', sum(test_auc_mean) / 5])
    csv_writer.writerow(['average_F1_score', sum(test_f1_score_mean) / 5])
    csv_writer.writerow(['average_ACC', sum(test_acc_mean) / 5])
    csv_writer.writerow(['average_Sen', sum(test_sen) / 5])
    csv_writer.writerow(['average_Spe', sum(test_spe) / 5])

    save_file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Features_train')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--study', type=str, default='brca')
    parser.add_argument('--features_path', type=str, default='/data/features/TCGA_BRCA/RetCCL/pt_files/')
    parser.add_argument('--feat_input', type=int, default=2048)
    parser.add_argument('--lrf', type=float, default=0.1, help='Learning Rate Final')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=5)
    # label file
    parser.add_argument('--csv_file', type=str, default='/M4_csv/10genes/all_brca.csv')
    parser.add_argument('--seed', type=int, default=42)
    # result dir
    parser.add_argument('--save_path', type=str, default='./results/BRCA/M4/')


    #Model related
    parser.add_argument('--model', type=str, default='AMIL', help='Type of model (Default: AMIL)')
    parser.add_argument('--num_expert', type=int, default=5)
    parser.add_argument('--experts_out', type=int, default=512, help='The output dimension of experts')
    parser.add_argument('--multi', action='store_true', help='IF --multi is in the scripts, the value is true.')
    
    opt = parser.parse_args()
    main(opt)
