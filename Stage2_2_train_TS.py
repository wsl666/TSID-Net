import argparse
import time
import numpy as np
from data.metrics import psnr,ssim
import warnings
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch import optim, nn
from torch.utils.data import DataLoader
import utils
from data.dataloader import TrainDataloader_SR,TestDataloader
from networks import VGG19CLCR,TSNet,VGG19CR
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Basic options
parser = argparse.ArgumentParser()
parser.add_argument('--clear_dir', type=str, default='./datasets/BeDDE/train/clear/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--hazy_dir', type=str, default='./results/stylized_BeDDE/stylized200/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--category', type=str, default='BeDDE',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--stylized_dir', type=str, default='./results/stylized_BeDDE/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--hazy_test_dir', type=str, default='./datasets/BeDDE/test/haze/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--clear_test_dir', type=str, default='./datasets/BeDDE/test/clear/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--save_model_dir', default='./checkpoints/BeDDE/',help='Directory to save the model')
parser.add_argument('--save_val_dir', default='./results/val/',help='Directory to save the val results')
parser.add_argument('--save_good_model_dir', default='./checkpoints/GoodT/BeDDE_T.pth',help='Directory to save the model')
parser.add_argument('--save_bad_model_dir', default='./checkpoints/BadT/BeDDE_T.pth',help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',help='Directory to save the logs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, default=50)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gps', type=int, default=3)
parser.add_argument('--blocks', type=int, default=6)
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--cl_lambda', type=float, default=0.25)
parser.add_argument('--clcr_weight', type=float, default=0.1)
parser.add_argument('--kd_weight', type=float, default=0.1)
parser.add_argument('--n_threads', type=int, default=0)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
    ]

    def curriculum_weight(difficulty):

        diff_list = [18, 20, 25, 27, 32]
        # diff_list = [22, 24, 26, 28, 30]
        weights = [(1 + args.cl_lambda) if difficulty > x else (1 - args.cl_lambda) for x in diff_list]
        weights.append(len(diff_list))
        new_weights = [i / sum(weights) for i in weights]

        return new_weights

    train_dataset = TrainDataloader_SR(args.hazy_dir,args.clear_dir,args.stylized_dir,transform=transforms_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    val_sets = TestDataloader(args.hazy_test_dir, args.clear_test_dir, transform=transforms_train)
    val_loader = DataLoader(dataset=val_sets, batch_size=args.batch_size//args.batch_size, shuffle=False)

    dataset_length = len(train_loader)

    logger_train=utils.Logger(args.max_epoch,dataset_length)
    logger_val  = utils.Logger(args.max_epoch, len(val_loader))

    # S_R = SRNetPlus.SRNet(gps=args.gps,blocks=args.blocks).to(device) # Stylized to Real
    T_S = TSNet.TSNet(gps=args.gps,blocks=args.blocks).to(device) # Stylized to Real


    print('The models are initialized successfully!')

    T_S.train()

    opt_T_S = optim.Adam(T_S.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_T_S = torch.optim.lr_scheduler.LambdaLR(opt_T_S, lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch, args.decay_epoch).step)

    loss_l1 = nn.L1Loss().to(device)   # L1 loss
    loss_clcr = VGG19CLCR.ContrastLoss().to(device)
    loss_ckt = TSNet.CKTTeacher(goodt_path=args.save_good_model_dir,badt_path=args.save_bad_model_dir).to(device)

    total_params = sum(p.numel() for p in T_S.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    max_ssim = 0
    max_psnr = 0
    all_ssims = []
    all_psnrs = []

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        ssims = []  # 每轮清空
        psnrs = []  # 每轮清空

        if epoch == args.start_epoch:
            weights = curriculum_weight(0)
            print(
                f' n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|inp_weight:{weights[5]}')
        else:
            weights = curriculum_weight(max_psnr)
            print(
                f' max_psnr:{max_psnr}| n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|inp_weight:{weights[5]}')
        for i, batch in enumerate(train_loader):

            x = batch[0].to(device)   # selected hazy images
            clear = batch[1].to(device)         # clear images
            # image_name= batch[2]
            candidate_list = batch[3]

            output = T_S(x)

            loss_L1 = loss_l1(output[0], clear) * args.l1_weight
            loss_CLCR = loss_clcr(output[0], clear,x,candidate_list,weights) * args.clcr_weight
            loss_CKT = loss_ckt(output[1:], clear,x) * args.kd_weight

            loss = loss_L1 + loss_CLCR + loss_CKT
            # loss = loss_L1 + loss_CLCR

            opt_T_S.zero_grad()
            loss.backward()
            opt_T_S.step()

            logger_train.log_train({
                                    # 'loss_l1': loss_L1,
                                    # 'loss_CLCR': loss_CLCR,
                                    # 'loss_CKT': loss_CKT,
                                    'loss': loss},
                                   images={'Hazy': x, 'Clear': clear, 'Output': output[0]})

        lr_scheduler_T_S.step()

    ################################################ Validating ##########################################

        with torch.no_grad():

            T_S.eval()

            torch.cuda.empty_cache()

            images_val = []
            images_name = []
            print("epoch:{}---> Metrics are being evaluated！".format(epoch))

            for a, batch_val in enumerate(val_loader):

                haze_val  = batch_val[0].to(device)
                clear_val = batch_val[1].to(device)

                image_name= batch_val[2][0]

                output_val,_ = T_S(haze_val)

                images_val.append(output_val)
                images_name.append(image_name)

                psnr1 = psnr(output_val, clear_val)
                ssim1 = ssim(output_val, clear_val).item()

                psnrs.append(psnr1)
                ssims.append(ssim1)

                logger_val.log_val({'PSNR': psnr1,
                                    'SSIM': ssim1},
                                   images={'output_val': output_val, 'val': clear_val})

            psnr_eval = np.mean(psnrs)
            ssim_eval = np.mean(ssims)

            if psnr_eval > max_psnr:

                max_psnr = max(max_psnr, psnr_eval)

                torch.save(T_S.state_dict(), args.save_model_dir + args.category + "_Best_PSNR.pth")

                for i in range(len(images_name)):

                    torchvision.utils.save_image(images_val[i], args.save_val_dir + "{}".format(images_name[i]))

            if ssim_eval > max_ssim:

                max_ssim = max(max_ssim, ssim_eval)

                torch.save(T_S.state_dict(), args.save_model_dir + args.category + "_Best_SSIM.pth")


