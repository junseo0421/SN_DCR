import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
from data.utils import *
from torch.utils.data import Dataset
import data.dtype as dtype

import torchvision.transforms.functional as T_FUNC

def imwrite(image, path,gray,resize, **plugin_args):
    #normalize -1 to 1
    mins=np.min(image,axis=(0,1))
    maxs=np.max(image,axis=(0,1))
    for i in range(image.shape[2]):
        if mins[i]<-1 or maxs[i]>1:
            image[:,:,i]=(((image[:,:,i]-mins[i])/(maxs[i]-mins[i]))*2)-1
    """Save a [-1.0, 1.0] image."""
    if gray==True:
        image=icol.rgb2gray(image)
    if resize==True:
        image=skiT.resize(image,(70,180))
    iio.imsave(path, dtype.im2uint(image), **plugin_args)

# class CUSTOMDataset(BaseDataset):
#     """
#     This dataset class can load unaligned/unpaired datasets.
#
#     It requires two directories to host training images from domain A '/path/to/data/trainA'
#     and from domain B '/path/to/data/trainB' respectively.
#     You can train the model with the dataset flag '--dataroot /path/to/data'.
#     Similarly, you need to prepare two directories:
#     '/path/to/data/testA' and '/path/to/data/testB' during test time.
#     """
#
#     def __init__(self, opt):
#         """Initialize this dataset class.
#
#         Parameters:
#             opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseDataset.__init__(self, opt)
#
#         self.A_paths = glob(opt.dataroot, '*/*', True)   # load images from '/path/to/data/trainA'
#
#
#     def __getitem__(self, index):
#         A_img = Image.open(self.A_paths[index]).convert('RGB')
#
#
#         is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
#         modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
#         transform = get_transform(modified_opt)
#
#         A = transform(A_img)
#
#         img2 = torch.zeros(size=(3,256,256))
#
#         img2[:, :,42:214]  = A[:, :, 42:214]
#
#         return {'A': img2, 'B': A, 'A_paths': self.A_paths[index], 'B_paths': self.A_paths[index]}
#
#     def __len__(self):
#         """Return the total number of images in the dataset.
#
#         As we have two datasets with potentially different number of images,
#         we take a maximum of
#         """
#         return len(self.A_paths)
class CUSTOMDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.A_paths = glob(opt.dataroot, '*/*', True)   # load images from '/path/to/data/trainA'


    def __getitem__(self, index):
        A_img = Image.open(self.A_paths[index]).convert('RGB')


        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        A = transform(A_img)

        img2 = torch.zeros(size=(3,256,256))

        img2[:, :, 71:185] = A[:, :, 71:185]

        return {'A': img2, 'B': A, 'A_paths': self.A_paths[index], 'B_paths': self.A_paths[index]}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.A_paths)
        
class FingerveinDataset_TEST(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.A_paths = glob(opt.dataroot, '*/*', True)  # load images from '/path/to/data/trainA'
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.A_paths)
        # return 10000

    def __getitem__(self, index):
        A_img = Image.open(self.A_paths[index]).convert('RGB')
        A_img = A_img.resize((114, 256))

        A = self.transform(A_img)

        img2 = torch.zeros(size=(3,256,256))

        img2[:, :, 71:185] = A

        A = T_FUNC.resize(A,(256,256))
        # return {'A': torch.rand((3,256,256)), 'B': torch.rand((3,256,256)),'A_paths': '', 'B_paths': ''}
        return {'A': img2, 'B': A,'A_paths': self.A_paths[index], 'B_paths': self.A_paths[index]}