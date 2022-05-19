import torch
import torchvision
from utils.utils import data_prefetcher, cal_normfam, setup_seed
from pretrainedmodels import xception
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils.datasets_profiles as dp

resume_model = ""

aug = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])])


def gen_heatmap(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    fam = heatmap + np.float32(image)
    return norm_image(fam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    image = image.copy()
    image -= np.min(image)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


if __name__ == "__main__":
    setup_seed(0)
    batch_size = 32
    num_workers = 0
    pin_memory = False
    cuda_device = "cuda:0"
    torch.cuda.set_device(cuda_device)

    model = xception(num_classes=2, pretrained=False).cuda()

    model.load_state_dict(torch.load(resume_model, map_location={'cuda:0': cuda_device, 'cuda:1': cuda_device, 'cuda:2': cuda_device, 'cuda:3': cuda_device}))

    dataset = dp.DFFD()
    testset, dffddir = dataset.getsetlist(False, 2)

    testdataloaders = []
    for s in testset:
        testdataloaders.append(DataLoader(s,
                                          batch_size=batch_size,
                                          pin_memory=pin_memory,
                                          num_workers=num_workers,
                                          shuffle=True))

    print("Loaded model")

    model.eval()

    ind = 0
    Dict = {}
    for loader in testdataloaders:
        prefetcher = data_prefetcher(loader)
        data, y_true = prefetcher.next()

        sum_map = np.zeros((224, 224))
        sum_data = np.zeros((224, 224, 3))

        cnt = 0
        while data is not None:
            tmpdata = data.clone().cpu().numpy()
            for i in range(len(tmpdata)):
                if torch.rand(1) < 0.5:
                    tmpdata[i] = np.flip(tmpdata[i], -1)
            sum_data += np.average(tmpdata, axis=0).transpose((1, 2, 0))

            fam = cal_normfam(model, data)
            fam = fam.cpu().detach().numpy()
            for i in range(len(fam)):
                if torch.rand(1) < 0.5:
                    fam[i] = np.flip(fam[i], -1)
            fam = np.mean(fam, axis=0)[0]
            sum_map += fam
            cnt += 1
            if cnt >= 8:
                break
            data, y_true = prefetcher.next()

        sum_map = sum_map/cnt
        sum_map -= np.min(sum_map)
        sum_map /= np.max(sum_map)
        sum_data = sum_data/cnt
        sum_data = sum_data / 4 + 0.5

        cam, heatmap = gen_heatmap(sum_data, sum_map)
        Dict[dffddir[ind]] = [sum_data, sum_map, cam, heatmap]
        ind += 1

    size = (3, 3)
    for i in range(4):
        plt.figure(dpi=400)
        ind = 0
        for x, y in Dict.items():
            ind += 1
            plt.subplot(*size, ind)
            if i == 1:
                X, Y = np.meshgrid(np.arange(0, 224, 1), np.arange(224, 0, -1))
                cset = plt.contourf(X, Y, y[i], 20)
                plt.colorbar(cset)
            else:
                plt.imshow(y[i])
            plt.xticks(())
            plt.yticks(())
            plt.title(x)
        plt.savefig("../"+str(i)+".png")
