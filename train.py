import os
import model
import json
import torch
import pickle
import mlflow
import argparse
from data import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn import functional as F


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocessing script")
    parser.add_argument("-d", "--data", required=True,
                        help="Path to preprocessing output directory.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output directory.")
    parser.add_argument("--experiment-id", required=False, 
                        default="U-Net Lung Segmentation 1",
                        help="Path to output directory.")
    parser.add_argument("-m", "--mode", required=False, default="3d",
                        help="2d or 3d.")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--learning-rate", required=False, type=float,
                        default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", required=False, type=int, default=1,
                        help="Batch size.")
    parser.add_argument("--train-size", required=False, type=int, default=800,
                        help="Size of train set.")
    parser.add_argument("--validation-size", required=False, type=int, default=50,
                        help="Size of validation set.")
    parser.add_argument("--validation-steps", required=False, type=int, default=200,
                        help="Number of batches until validation.")
    parser.add_argument("--start-filters", required=False, type=int, default=32,
                        help="Start filters.")
    parser.add_argument("--n-classes", required=False, type=int, default=1,
                        help="Number of output classes.")
    parser.add_argument("-x", "--scan-size-x", required=False, type=int, default=128,
                        help="X axis size of preprocessed CT scans.")
    parser.add_argument("-y", "--scan-size-y", required=False, type=int, default=256,
                        help="Y axis size of preprocessed CT scans.")
    parser.add_argument("-z", "--scan-size-z", required=False, type=int, default=256,
                        help="Z axis size of preprocessed CT scans.")
    parser.add_argument("--scans-per-batch", required=False, type=int, default=1,
                        help="Number of scans per batch.")
    parser.add_argument("--slices-per-batch", required=False, type=int, default=4,
                        help="Number of slices per batch.")
    parser.add_argument("--neg-examples-per-batch", required=False, type=int, default=0,
                        help="Number of negative examples per batch.")
    args = parser.parse_args()
    args.scan_size = np.array([args.scan_size_x, args.scan_size_y, args.scan_size_z])

    # CUDA setup
    torch.cuda.set_enabled_lms(True)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    # MLFlow setup
    remote_server_uri = "http://mlflow.10.7.13.202.nip.io/"
    sftp_uri = "sftp://mlflow:mlflow@mlflow-mop.mlflow.svc.cluster.local/mlflow_data/artifacts"
    mlflow.set_tracking_uri(remote_server_uri)
    try:
        exp_id = mlflow.create_experiment(args.experiment_id, artifact_location=sftp_uri)
    except:
        exp_id = mlflow.set_experiment(args.experiment_id)

    # Create output dir if needed
    if not os.path.exists(os.path.join(args.output, "model")):
        os.makedirs(os.path.join(args.output, "model"))

    # Scan list
    st_scans = np.load(os.path.join(args.data, "list_scans.npy"))
    
    if args.mode == "3d":
        train_scans = st_scans[:args.train_size]
        val_scans = st_scans[args.train_size:]
        train_data = dataset.Dataset(train_scans, args.data,
                                    mode="3d", scan_size=args.scan_size, n_classes=args.n_classes)
        val_data = dataset.Dataset(val_scans, args.data,
                                    mode="3d", scan_size=args.scan_size)
        unet = model.UNet(1, args.n_classes,
                        args.start_filters, bilinear=False).cuda()
        criterion = utils.dice_loss
        optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)
        batch_size = args.batch_size
        epochs = args.epochs
        val_steps = args.validation_steps
        val_size = args.validation_size
    else:
        st_scans = st_scans[:args.train_size]
        dataset = dataset.Dataset(
            st_scans, args.data, mode="2d")
        unet = model.UNet(1, 1, args.start_filters, bilinear=True).to(device)
        criterion = utils.dice_loss
        optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)
        batch_size = args.batch_size
        slices_per_batch = args.slices_per_batch
        neg = args.neg_examples_per_batch
        epochs = args.epochs

    best_val_loss = 1e16

    with mlflow.start_run(experiment_id=exp_id) as run:
        # Log hyper-params to MLFlow
        mlflow.log_param("Learning rate", args.learning_rate)
        mlflow.log_param("Epochs", args.epochs)
        mlflow.log_param("Batch size", args.batch_size)
        mlflow.log_param("Train set", args.train_size)
        mlflow.log_param("Validation set", args.validation_size)
        mlflow.log_param("Scan size", args.scan_size)
        val_loss_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch_loss = 0
                batch = np.array([train_data[j][0]
                                for j in range(i, i+batch_size)]).astype(np.float16)
                labels = np.array([train_data[j][1]
                                for j in range(i, i+batch_size)]).astype(np.float16)

                batch = torch.Tensor(batch).to(device)
                labels = torch.Tensor(labels).to(device)
                batch.requires_grad = True
                labels.requires_grad = True

                optimizer.zero_grad()
                logits = unet(batch).cuda()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                print("Epoch {} ==> Batch {} mean loss : {}".format(
                    epoch+1, (i+1) % (val_steps), loss.item()/batch_size))
                epoch_loss += loss.item()/batch_size
                del batch
                del labels
                torch.cuda.empty_cache()
                if (i+1) % val_steps == 0:
                    print("===================> Calculating validation loss ... ")
                    ids = np.random.randint(0, len(val_data), val_size)
                    val_loss = 0
                    for scan_id in ids:
                        batch = np.array([val_data[j][0] for j in range(
                            scan_id, scan_id+batch_size)]).astype(np.float16)
                        labels = np.array([val_data[j][1] for j in range(
                            scan_id, scan_id+batch_size)]).astype(np.float16)
                        batch = torch.Tensor(batch).to(device)
                        labels = torch.Tensor(labels).to(device)
                        logits = unet(batch)
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                    val_loss /= val_size

                    # Log mean loss to MLFLow
                    val_loss_step = val_loss_step + 1
                    mlflow.log_metric("Validation dice loss", val_loss, step=val_loss_step)
                
                    print("\n # Validation Loss : ", val_loss)
                    if val_loss < best_val_loss:
                        print("\nSaving Better Model... ")
                        torch.save(unet.state_dict(), os.path.join(args.output, "model", "model"))
                        best_val_loss = val_loss
                    print("\n")
        print("\nUploading best model to MLFlow...")
        mlflow.pytorch.log_model(unet.load_state_dict(torch.load(os.path.join(args.output, "model", "model"))), "models")

