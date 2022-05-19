import torch
from utils.utils import data_prefetcher_two, cal_fam, setup_seed, calRes
from pretrainedmodels import xception
import utils.datasets_profiles as dp
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import argparse
import random
import time

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()

parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--modelname', default="xception", type=str)
parser.add_argument('--distributed', default=False, action='store_true')
parser.add_argument('--upper', default="xbase", type=str,
                    help='the prefix used in save files')

parser.add_argument('--eH', default=120, type=int)
parser.add_argument('--eW', default=120, type=int)

parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_batch', default=500000, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--logbatch', default=3000, type=int)
parser.add_argument('--savebatch', default=30000, type=int)
parser.add_argument('--seed', default=5, type=int)

parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')

parser.add_argument('--pin_memory', '-p', default=False, action='store_true')
parser.add_argument('--resume_model', default=None)
parser.add_argument('--resume_optim', default=None)

parser.add_argument('--save_model', default=True, action='store_true')
parser.add_argument('--save_optim', default=False, action='store_true')

args = parser.parse_args()
upper = args.upper


def Eval(model, lossfunc, dtloader):
    model.eval()
    sumloss = 0.
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j, batch) in enumerate(dtloader):
            x, y_true = batch
            y_pred = model.forward(x.cuda())

            loss = lossfunc(y_pred, y_true.cuda())
            sumloss += loss.detach()*len(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))

    return sumloss/len(y_true_all), y_true_all.detach(), y_pred_all.detach()


def Log(log):
    print(log)
    f = open("./logs/"+upper+"_"+modelname+".log", "a")
    f.write(log+"\n")
    f.close()


if __name__ == "__main__":
    Log("\nModel:%s BatchSize:%d lr:%f" % (modelname, args.batch_size, args.lr))
    torch.cuda.set_device(args.device)
    setup_seed(args.seed)
    print("cudnn.version:%s enabled:%s benchmark:%s deterministic:%s" % (torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))

    MAX_TPR_4 = 0.

    model = xception(num_classes=2, pretrained=False).cuda()

    if args.distributed:
        model = torch.nn.DataParallel(model)

    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if args.resume_model is not None:
        model.load_state_dict(torch.load(args.resume_model))
    if args.resume_optim is not None:
        optim.load_state_dict(torch.load(args.resume_optim))

    lossfunc = torch.nn.CrossEntropyLoss()

    dataset = dp.DFFD()

    trainsetR = dataset.getTrainsetR()
    trainsetF = dataset.getTrainsetF()

    validset = dataset.getValidset()

    testsetR = dataset.getTestsetR()
    TestsetList, TestsetName = dataset.getsetlist(real=False, setType=2)

    setup_seed(args.seed)

    traindataloaderR = DataLoader(
        trainsetR,
        batch_size=int(args.batch_size/2),
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    traindataloaderF = DataLoader(
        trainsetF,
        batch_size=int(args.batch_size/2),
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    validdataloader = DataLoader(
        validset,
        batch_size=args.batch_size*2,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    testdataloaderR = DataLoader(
        testsetR,
        batch_size=args.batch_size*2,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    testdataloaderList = []
    for tmptestset in TestsetList:
        testdataloaderList.append(
            DataLoader(
                tmptestset,
                batch_size=args.batch_size*2,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers
            )
        )

    print("Loaded model")

    batchind = 0
    e = 0
    sumcnt = 0
    sumloss = 0.
    while True:
        prefetcher = data_prefetcher_two(traindataloaderR, traindataloaderF)

        data, y_true = prefetcher.next()

        while data is not None and batchind < args.max_batch:
            stime = time.time()
            sumcnt += len(data)

            ''' ↓ the implementation of RFM ↓ '''
            model.eval()
            mask = cal_fam(model, data)
            imgmask = torch.ones_like(mask)
            imgh = imgw = 224

            for i in range(len(mask)):
                maxind = np.argsort(mask[i].cpu().numpy().flatten())[::-1]
                pointcnt = 0
                for pointind in maxind:
                    pointx = pointind//imgw
                    pointy = pointind % imgw

                    if imgmask[i][0][pointx][pointy] == 1:

                        maskh = random.randint(1, args.eH)
                        maskw = random.randint(1, args.eW)

                        sh = random.randint(1, maskh)
                        sw = random.randint(1, maskw)

                        top = max(pointx-sh, 0)
                        bot = min(pointx+(maskh-sh), imgh)
                        lef = max(pointy-sw, 0)
                        rig = min(pointy+(maskw-sw), imgw)

                        imgmask[i][:, top:bot, lef:rig] = torch.zeros_like(imgmask[i][:, top:bot, lef:rig])

                        pointcnt += 1
                        if pointcnt >= 3:
                            break

            data = imgmask * data + (1-imgmask) * (torch.rand_like(data)*2-1.)
            ''' ↑ the implementation of RFM ↑ '''

            model.train()
            y_pred = model.forward(data)
            loss = lossfunc(y_pred, y_true)

            flood = (loss-0.04).abs() + 0.04
            sumloss += loss.detach()*len(data)
            data, y_true = prefetcher.next()

            optim.zero_grad()
            flood.backward()
            optim.step()

            batchind += 1
            print("Train %06d loss:%.5f avgloss:%.5f lr:%.6f time:%.4f" % (batchind, loss, sumloss/sumcnt, optim.param_groups[0]["lr"], time.time()-stime), end="\r")

            if batchind % args.logbatch == 0:
                print()
                Log("epoch:%03d batch:%06d loss:%.5f avgloss:%.5f" % (e, batchind, loss, sumloss/sumcnt))

                loss_valid, y_true_valid, y_pred_valid = Eval(model, lossfunc, validdataloader)
                ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(y_true_valid, y_pred_valid)
                Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s" % (AUC, TPR_2, TPR_3, TPR_4, "validset"))

                loss_r, y_true_r, y_pred_r = Eval(model, lossfunc, testdataloaderR)
                sumAUC = sumTPR_2 = sumTPR_3 = sumTPR_4 = 0
                for i, tmptestdataloader in enumerate(testdataloaderList):
                    loss_f, y_true_f, y_pred_f = Eval(model, lossfunc, tmptestdataloader)
                    ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(torch.cat((y_true_r, y_true_f)), torch.cat((y_pred_r, y_pred_f)))
                    sumAUC += AUC
                    sumTPR_2 += TPR_2
                    sumTPR_3 += TPR_3
                    sumTPR_4 += TPR_4
                    Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s" % (AUC, TPR_2, TPR_3, TPR_4, TestsetName[i]))
                if len(testdataloaderList) > 1:
                    Log("AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f Test" %
                        (sumAUC/len(testdataloaderList), sumTPR_2/len(testdataloaderList), sumTPR_3/len(testdataloaderList), sumTPR_4/len(testdataloaderList)))
                    TPR_4 = (sumTPR_4)/len(testdataloaderList)

                if batchind % args.savebatch == 0 or TPR_4 > MAX_TPR_4:
                    MAX_TPR_4 = TPR_4
                    if args.save_model:
                        torch.save(model.state_dict(), "./models/" + upper+"_"+modelname+"_model_batch_"+str(batchind))
                    if args.save_optim:
                        torch.save(optim.state_dict(), "./models/" + upper+"_"+modelname+"_optim_batch_"+str(batchind))

                print("-------------------------------------------")
        e += 1
