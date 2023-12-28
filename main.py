import warnings
from data.metrics import ssim,psnr
import torchvision
from torch.utils.data import DataLoader
from data.dataloader  import TestDataloader
import torch
import numpy as np
from torchvision import transforms as tf
from PIL import Image
import torch.nn.functional as F

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dehaze_path = 'datasets/BeDDE/train/haze/'  # 有雾图像保存路径
clear_path = 'results/stylized200/'  # 清晰图像保存路径

if __name__ == "__main__":

    transforms_test = [
        tf.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ]

    test_sets = TestDataloader(dehaze_path, clear_path,transform=transforms_test)

    dataloader=DataLoader(dataset=test_sets,batch_size=1,shuffle=False,num_workers=4)

    psnrs=[]
    ssims=[]
    errors=[]
    list_score_h=[]
    list_score_c=[]

    # iqa_metric = pyiqa.create_metric('musiq', as_loss=True).cuda()
    #
    # net= GetGradientNopadding().to(device)
    # q=0
    for a, batch in enumerate(dataloader):

        dehaze = batch[0].to(device)
        clear = batch[1].to(device)

        # h=clear.shape[2]
        # w=clear.shape[3]
        #

        # score_h = iqa_metric(dehaze).detach().cpu().numpy()
        # score_c = iqa_metric(clear).detach().cpu().numpy()
        # print(score_h,score_c)
        # image_name = batch[2][0]
        #
        # haze_grad=net(dehaze).to(device)
        # clear_grad=net(clear).to(device)
        # error=dehaze-haze_grad
        # resized1 = F.interpolate(dehaze, size=(576, 576), mode='bicubic') #还原回原尺寸大小
        # resized2 = F.interpolate(clear, size=(576, 576), mode='bicubic') #还原回原尺寸大小
        #
        # resized3 = F.interpolate(dehaze, size=(h, w), mode='bicubic') #还原回原尺寸大小
        # #
        # torchvision.utils.save_image(haze_grad, "results2/haze/{}".format(image_name))
        # torchvision.utils.save_image(clear_grad, "results2/clear/{}".format(image_name))
        # torchvision.utils.save_image(dehaze-haze_grad, "results2/h/{}".format(image_name))
        #
        psnrs.append(psnr(dehaze,clear))
        ssims.append(ssim(dehaze,clear).item())
        #
        # if score_h> score_c:
        #     q +=1

        print(psnr(dehaze,clear),ssim(dehaze,clear).item())
    # print((len(dataloader)-q)/len(dataloader))
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    #
    print("avg_psnr：{}".format(avg_psnr))
    print("avg_ssim：{}".format(avg_ssim))

