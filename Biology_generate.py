import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm

import config
import utils

import os
from skimage import io
import tifffile as tiff

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

##### Build Model #####
if args.model.lower() == 'cain_encdec':
    from model.cain_encdec import CAIN_EncDec
    print('Building model: CAIN_EncDec')
    model = CAIN_EncDec(depth=args.depth, start_filts=32)
elif args.model.lower() == 'cain':
    from model.cain import CAIN
    print("Building model: CAIN")
    model = CAIN(depth=args.depth)
elif args.model.lower() == 'cain_noca':
    from model.cain_noca import CAIN_NoCA
    print("Building model: CAIN_NoCA")
    model = CAIN_NoCA(depth=args.depth)
else:
    raise NotImplementedError("Unknown model!")
# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
#print(model)

print('# of parameters: %d' % sum(p.numel() for p in model.parameters()))


# If resume, load checkpoint: model
if args.resume:
    #utils.load_checkpoint(args, model, optimizer=None)
    load_name = os.path.join('checkpoint', args.resume_exp, 'model_best.pth')
    checkpoint = torch.load(load_name)
    print("loading checkpoint %s" % load_name) 
    #checkpoint = torch.load('pretrained_cain.pth')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint


def test(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    model.eval()

    for i in range(1,14):
        w1 = '20230617-U2OS-NileRed-%0.3d_w1CSU GFP.TIF' % i
        w2 = '20230617-U2OS-NileRed-%0.3d_w2CSU LMQ B2G.TIF' % i
        w3 = '20230617-U2OS-NileRed-%0.3d_w3CSU LMQ-B2R.TIF' % i

        w1_img = io.imread(os.path.join(args.data_root,w1)).astype(np.float32)
        w2_img = io.imread(os.path.join(args.data_root,w2)).astype(np.float32)
        w3_img = io.imread(os.path.join(args.data_root,w3)).astype(np.float32)

        w1_img = torch.Tensor(w1_img)
        w2_img = torch.Tensor(w2_img)
        w3_img = torch.Tensor(w3_img)

        result_img_w1 = w1_img[0].unsqueeze(dim=0)
        result_img_w2 = w2_img[0].unsqueeze(dim=0)
        result_img_w3 = w3_img[0].unsqueeze(dim=0)

        for j in range(w1_img.shape[0] - 1):
            img1 = torch.stack((w1_img[j], w2_img[j], w3_img[j]), dim=0)
            img2 = torch.stack((w1_img[j+1], w2_img[j+1], w3_img[j+1]), dim=0)

            
            im1 = img1.unsqueeze(dim=0)
            im2 = img2.unsqueeze(dim=0)
            
            im1 = im1.to(device)
            im2 = im2.to(device)
            print(im1.shape)

            out, _ = model(im1, im2)
            out = out[0].to('cpu')

            result_img_w1 = torch.cat((result_img_w1, out[0].unsqueeze(0), w1_img[j+1].unsqueeze(0)), dim=0)
            result_img_w2 = torch.cat((result_img_w2, out[0].unsqueeze(0), w1_img[j+1].unsqueeze(0)), dim=0)
            result_img_w3 = torch.cat((result_img_w3, out[0].unsqueeze(0), w1_img[j+1].unsqueeze(0)), dim=0)

        tiff.imwrite("test/w1.TIF", result_img_w1)
        tiff.imwrite("test/w2.TIF", result_img_w2)
        tiff.imwrite("test/w3.TIF", result_img_w3)

        break
            
    return


""" Entry Point """
def main(args):
    # num_iter = 1 # x2**num_iter interpolation
    # for _ in range(num_iter):
        
    #     # run test
    #     test(args, args.start_epoch)

    test(args, args.start_epoch)


if __name__ == "__main__":
    main(args)
