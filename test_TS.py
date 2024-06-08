from networks import TSNet
from torchvision import transforms as tf
import torch.nn.functional as F
import argparse
import time
import numpy as np
from data.metrics import psnr,ssim
import warnings
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch.utils.data import DataLoader
import utils
from data.dataloader import TestDataloader
from PIL import Image
from PIL import ImageFile

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None           # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--hazy_dir', type=str, default='./datasets/real-images/',help='Directory path to a batch of clear images')
parser.add_argument('--clear_dir', type=str, default='./datasets/real-images/',help='Directory path to a batch of hazy images')
parser.add_argument('--save_model_dir', default='./checkpoints/outdoor/',help='Directory to save the teacher model')
parser.add_argument('--save_val_dir', default='./results/test_results/',help='Directory to save the teacher model')
parser.add_argument('--max_epoch', type=int, default=1)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--gps', type=int, default=3)
parser.add_argument('--blocks', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=0)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_test = [
        # tf.Resize((args.load_size,args.load_size), Image.BICUBIC),
        tf.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]

    test_sets = TestDataloader(args.hazy_dir, args.hazy_dir, transform=transforms_test)

    dataloader=DataLoader(dataset=test_sets,batch_size=args.batch_size,shuffle=False,num_workers=args.n_threads)

    T_S = TSNet.TSNet(gps=args.gps,blocks=args.blocks).to(device)

    T_S.load_state_dict(torch.load(args.save_model_dir + 'outdoor_Best_PSNR.pth'))
    # S_R.load_state_dict(torch.load(args.save_model_dir + 'G_Best_PSNR.pth'))

    T_S.eval()

    psnrs=[]
    ssims=[]

    print("###################### 模型加载成功！##########################")
    start_time = time.time()
    for a, batch in enumerate(dataloader):

        haze = batch[0].to(device)
        clear = batch[1].to(device)

        image_name=batch[2][0]
        size = batch[3]

        with torch.no_grad():

            output,_ = T_S(haze)

            psnrs.append(psnr(output, clear))
            ssims.append(ssim(output, clear).item())

            w = haze.shape[3]//2

            output = torch.cat([haze[:,:,:,0:w],output[:,:,:,w:w*2]],dim=3)
            # output = torch.cat([haze,output],dim=3)

            torchvision.utils.save_image(output, args.save_val_dir + "{}".format(image_name))
            # torchvision.utils.save_image(haze,  "./datasets/real-images/rename/GT_{}.png".format(a+93))

            print("第{}张,{}处理完成！".format(a+1,image_name))

    test_time = time.time() - start_time

    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)

    print("avg_psnr：{}".format(avg_psnr))
    print("avg_ssim：{}".format(avg_ssim))
    print("avg_time：{}s".format(test_time/len(dataloader)))

