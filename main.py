import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch.optim as optim
from get_filters.FixedScale_and_AdaptiveMultiScaleGaborFilters import FixedScale_and_AdaptiveMultiScaleGaborFilters
from GaborNet import GaborNet
import CTorchClasses as C
from create_Dataset import PalmprintTrainDataset_TripletMarginLoss, PalmprintTestDataset_TripletMarginLoss
# from torcheval.metrics.functional import binary_f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

class ConvolutionConfig:
    def __init__(self, DEVICE, imagedir_path, n_images, n_steps, countW, nbins, S, F, M, A_dash, r, a0, h1, h2, sigma, lambd, gamma=1, psi=0):

        self.DEVICE = DEVICE
        self.imagedir_path = imagedir_path
        self.n_images = n_images
        self.n_steps = n_steps
        self.countW = countW
        self.nbins = nbins
        self.S = S
        self.F = F
        self.M = M
        self.A_dash = A_dash
        self.r = r
        self.a0 = a0
        self.h1 = h1
        self.h2 = h2
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi


def train(epoch:int, model, train_dataloader, optimizer, loss_fn, DEVICE, writer, args):
    print("training...")
    model.train()
    avg_train_loss = 0
    
    with tqdm(train_dataloader, unit=" batch") as tepoch:
        for batch_id, (A, P, N) in enumerate(tepoch):
            
            tepoch.set_description(f"Epoch {epoch}")

            A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)
            anchor = model(A)
            positive = model(P)
            negative = model(N)
            loss = loss_fn(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix({"Loss": loss.item()})
            
            avg_train_loss += loss
    
    avg_train_loss /= len(train_dataloader)
    print("AVG TRAIN LOSS:", avg_train_loss.item(), "\n")
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
            

def test(epoch, model, test_dataloader, best_loss, DEVICE, MODEL_SAVE_PATH, writer, args):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix_target = torch.zeros(0)
    confusion_matrix_pred = torch.zeros(0)

    with torch.inference_mode():
        print("testing ...")
        for (images_1, images_2, targets) in tqdm(test_dataloader, unit="batch"):

            images_1, images_2, targets = images_1.to(DEVICE), images_2.to(DEVICE), targets.to(DEVICE)
            output_1 = model(images_1)
            output_2 = model(images_2)

            obj = C.TripletMarginWithComplexDistanceLoss(dist=args.dist_fn_method)
            outputs = obj.distance_function(output_1, output_2)
            outputs = outputs if args.dist_fn_method == "cosine_l2_norm" else outputs.abs()
            
            test_loss += torch.nn.functional.binary_cross_entropy(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs >= 0.3, 1, 0) # if distance greater than 0.3, not a match
            correct += pred.eq(targets.view_as(pred)).sum().item()

            confusion_matrix_target = torch.cat((confusion_matrix_target, targets.detach().cpu()), 0)
            confusion_matrix_pred = torch.cat((confusion_matrix_pred, pred.detach().cpu()), 0)

        test_loss /= len(test_dataloader)
        if test_loss < best_loss:
            print("saving best model...")
            torch.save(model, MODEL_SAVE_PATH)
            best_loss = test_loss
            print("creating confusion matrix...")
            match = re.search(r'/(\w+)$', args.model_save_path)
            cm = confusion_matrix(confusion_matrix_target, confusion_matrix_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=["match", "no_match"])
            disp.plot()
            plt.savefig(f"confusion_matrix/{match.group(1)}_confusion_matrix.png")
            print(cm)
        
        writer.add_scalar("Loss/test", test_loss, epoch)
        
        confusion_matrix_target = confusion_matrix_target.detach().cpu()
        confusion_matrix_pred = confusion_matrix_pred.detach().cpu()

        f1_score = fbeta_score(y_true=confusion_matrix_target, y_pred=confusion_matrix_pred, beta=1)

        writer.add_scalar("F1 Score", f1_score, epoch)
        
        f_beta_score = fbeta_score(y_true=confusion_matrix_target, y_pred=confusion_matrix_pred, beta=0.815)

        writer.add_scalar(f"FBETA Score (beta={args.f_beta})", f_beta_score, epoch)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))
        print("BEST LOSS:", best_loss)
        print("F1 SCORE:", f1_score.item())
        print(f"FBETA Score (beta={args.f_beta})", f_beta_score.item(), "\n")
        
    return best_loss

def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument("--train_imagedir_path", help="directory of train dataset", type=str)
    parser.add_argument("--test_imagedir_path", help="directory of test dataset", type=str)
    parser.add_argument("--n_images", help="number of images for obtaining the best wavelets", type=int)
    parser.add_argument("--n_steps", help="steps when slicing", type=int)
    parser.add_argument("--u,v", help="height and width of the image", type=int, nargs="+")
    parser.add_argument("--countW", help="countW", type=int)
    parser.add_argument("--nbins", help="number of bins", type=int)
    parser.add_argument("--F", help="countW", type=int)
    parser.add_argument("--h", help="countW", type=int)
    parser.add_argument("--lambd", help="countW", type=float)
    parser.add_argument("--sigma", help="countW", type=float)
    parser.add_argument("--A_dash", help="countW", type=int)
    parser.add_argument("--S", help="countW", type=int)
    parser.add_argument("--a0", help="countW", type=float)
    parser.add_argument("--r", help="countW", type=float)
    parser.add_argument("--BATCH", help="batch size for training", type=int)
    parser.add_argument("--dist_fn_method", help="distance function method for triplet loss", type=str)
    parser.add_argument("--f_beta", help="beta value for fbeta_score", type=float)
    parser.add_argument("--model_save_path", help="model save path", type=str)


    args = parser.parse_args()
    MODEL_SAVE_PATH = args.model_save_path
    
    time = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    match = re.search(r'/(\w+)$', args.model_save_path)
    log_dir = f"logruns/{time}_{match.group(1)}" if match else f"logruns/{time}_model"
    writer = SummaryWriter(log_dir=log_dir)
    
    u, v = getattr(args, 'u,v')

    config = ConvolutionConfig(DEVICE, imagedir_path="dataset/session2", n_images=args.n_images, n_steps=args.n_steps,
                                countW=args.countW, nbins=args.nbins, S=args.S, F=args.F, M=int(np.log2(u/2))-5, A_dash=args.A_dash,
                                r=args.r, a0=args.a0, h1=args.h, h2=args.h, sigma=args.sigma, lambd=args.lambd
                                )


    # IMAGES SHOULD COME FROM THE DATALOADER
    transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])]) # last one is transforms.ToTensor()
    train_img_dir = args.train_imagedir_path
    train_dataset = PalmprintTrainDataset_TripletMarginLoss(train_img_dir, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.BATCH, shuffle=True, drop_last=True)
    # A, P, N = next(iter(train_dataloader))

    test_img_dir = args.test_imagedir_path
    test_dataset = PalmprintTestDataset_TripletMarginLoss(test_img_dir, transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.BATCH, shuffle=True, drop_last=True)

    filters = FixedScale_and_AdaptiveMultiScaleGaborFilters(**vars(config))
    G_f = filters.compute_FixedScaleGaborFilters()
    G_A_dash = filters.compute_MultiScaleGaborFilters()

    net = GaborNet(torch.from_numpy(G_f), torch.from_numpy(G_A_dash))

    # net = nn.DataParallel(net, device_ids=[0,1,2])

    optimizer = optim.AdamW(net.parameters(), lr=0.5e-3)

    margin=1.
    loss_fn = C.TripletMarginWithComplexDistanceLoss(dist=args.dist_fn_method, margin=margin)

    EPOCH = 10
    best_loss = float('inf')
    for epoch in range(1, EPOCH+1):
        train(epoch=epoch, model=net, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, DEVICE=DEVICE, writer=writer, args=args)
        best_loss = test(epoch, net, test_dataloader, best_loss=best_loss, DEVICE=DEVICE, MODEL_SAVE_PATH=MODEL_SAVE_PATH, writer=writer, args=args)
    
    writer.flush()

if __name__ == "__main__":
    main()
