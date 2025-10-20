import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from data.custom_datasets import *
from util import html
import util.util as util
from torch.utils.data import DataLoader

from ptflops import get_model_complexity_info

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    ds = FingerveinDataset_TEST(opt)
    dataset = DataLoader(ds, batch_size=1, shuffle=True)

    model = create_model(opt)      # create a model given opt.model and other options
    model.eval()
    
    ## flops 측정용
    # cycle GAN model.netG_A
    # pix2pix GAN model.netG
    # macs, params = get_model_complexity_info(model.netG, (3, 256, 256), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    #
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # ## flops 측정용


    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        if i == 0:
            model.data_dependent_initialize()
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.eval()

        paths = data['A_paths'][0].split('/')
        # start=time.time()

        model.test()           # run inference

        # # 프로세싱 time 측정
        # print(time.time()-start)
        #
        # print(f"  Allocated: {round(torch.cuda.memory_allocated(0)/1024**2,1)} MB")
        # print(f"  Cached:    {round(torch.cuda.memory_cached(0)/1024**3,1)} GB\n")
        # # 프로세싱 time 측정
        
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        folderpath = opt.results_dir + '/' + paths[-2] + '/'

        # EPOCH 내의 클래스별로 이미지 생성
        mkdir(folderpath)

        # deblur image save
        g = (visuals['fake_B'][0].permute(1, 2, 0).cuda()).detach().cpu().numpy()

        imwrite(g, folderpath + paths[-1], gray=True, resize=True)