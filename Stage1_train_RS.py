import argparse
import sys
import warnings
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch import optim, nn
from torch.utils.data import DataLoader
import utils
from data.dataloader import TrainDataloader_RS
from networks import RSNet,Discriminator,loss_functions
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./datasets/BeDDE/train/clear/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--style_dir', type=str, default='./datasets/BeDDE/train/haze/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--category', type=str, default='BeDDE',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--save_model_dir', default='./checkpoints/',help='Directory to save the model')
parser.add_argument('--save_stylized_dir', default='./results/stylized_BeDDE/',help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',help='Directory to save the logs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--decay_epoch', type=int, default=100)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--identity1_weight', type=float, default=50.0)
parser.add_argument('--identity2_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=5.0)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--save_list', type=list, default=[150,160,170,180,190,200])
parser.add_argument('--n_threads', type=int, default=0)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
    ]

    train_dataset = TrainDataloader_RS(style_path=args.style_dir,content_path=args.content_dir,transform=transforms_train,unaligned=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    dataset_length = len(train_loader)

    logger_train=utils.Logger(args.max_epoch,dataset_length)

    R_S = RSNet.RSNet().to(device) # Real to Stylized
    M_D = Discriminator.MultiDiscriminator().to(device)

    print('The models are initialized successfully!')

    R_S.train()
    M_D.train()

    total_params = sum(p.numel() for p in R_S.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    label_real = torch.ones([1], dtype=torch.float, requires_grad=False).to(device)
    label_fake = torch.zeros([1], dtype=torch.float, requires_grad=False).to(device)

    opt_R_S = optim.Adam(R_S.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_M_D = optim.Adam(M_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_R_S = torch.optim.lr_scheduler.LambdaLR(opt_R_S, lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch, args.decay_epoch).step)
    lr_scheduler_M_D = torch.optim.lr_scheduler.LambdaLR(opt_M_D, lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch, args.decay_epoch).step)

    loss_identity = nn.MSELoss().to(device)   # identity loss

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        ######################################################
        # content_images_ = content_images[1:]
        # content_images_ = torch.cat([content_images_, content_images[0:1]], 0)
        # content_images = torch.cat([content_images, content_images_], 0)
        # style_images = torch.cat([style_images, style_images], 0)
        ######################################################

        for i, batch in enumerate(train_loader):

            style = batch[0].to(device)   # hazy images
            content = batch[1].to(device) # clear images
            image_name= batch[2]

            out, out_feats, content_feats, style_feats, c_c, s_s, c_c_feats, s_s_feats=R_S(content,style)

            # train discriminator
            loss_gan_d = M_D.compute_loss(style, label_real) + M_D.compute_loss(out.detach(), label_fake)
            opt_M_D.zero_grad()
            loss_gan_d.backward()
            opt_M_D.step()

            # train generator
            loss_gan_g = M_D.compute_loss(out, label_real) * args.gan_weight

            loss_c =  loss_functions.calc_content_loss(out_feats, content_feats, norm = True) * args.content_weight
            loss_s = loss_functions.calc_style_loss(out_feats, style_feats) * args.style_weight

            loss_identity1 = loss_identity(c_c, content) + loss_identity(s_s, style)
            loss_identity2 = loss_functions.calc_content_loss(c_c_feats, content_feats) + loss_functions.calc_content_loss(s_s_feats, style_feats)

            # initialize and updata reliable_style_bank
            # if epoch <= args.decay_epoch:
            #     utils.initialize_reliable_style_bank(out,args.save_stylized_dir, image_name)
            # elif epoch > args.decay_epoch:
            #     utils.update_reliable_style_bank(out,style,content,args.save_stylized_dir,image_name)

            loss = loss_c + loss_s + loss_identity1 * args.identity1_weight + loss_identity2 * args.identity2_weight + loss_gan_g

            opt_R_S.zero_grad()
            loss.backward()
            opt_R_S.step()

            logger_train.log_train({
                                    'loss_c': loss_c,
                                    'loss_s': loss_s,
                                    # 'loss_identity1': loss_identity1,  # Identity loss
                                    # 'loss_identity2': loss_identity2,  # Contrast loss
                                    # # 'loss_per': loss_per,  # Perceptual loss
                                    # 'loss_gan_g': loss_gan_g,  # Perceptual loss
                                    # 'loss': loss
                                    },  # Total discriminator loss , multi-discriminator

                                   images={'Style': style, 'Content': content, 'Output': out})
            # sys.stdout.write(
            #     '\rEpoch %03d/%03d [%04d/%04d] -- Loss %.6f ' % (epoch, args.max_epoch, i + 1, dataset_length , loss.item()))

            # output = torch.cat([haze_style, clear_content, out], 2)
            if epoch in args.save_list:
                for i in range(out.shape[0]):
                    save_image(out[i],args.save_stylized_dir+"stylized{}/{}".format(epoch,image_name[i]))


        lr_scheduler_R_S.step()
        lr_scheduler_M_D.step()

    torch.save(R_S.state_dict(), args.save_model_dir + args.category + "R_S.pth")