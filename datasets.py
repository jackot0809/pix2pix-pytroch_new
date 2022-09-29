import glob
import random
import os
from os.path import exists

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.mode=mode

        # if(self.mode=='train'):
        #     self.files_C = sorted(glob.glob(os.path.join(root, '%s/C' % mode) + '/*.*'))
        if(self.mode=='test'):
            # self.files_A.sort(key=lambda x:int(x.split('/')[-1][:-4]))
            self.filename=[x.split('/')[-1][:-4] for x in  self.files_A]
            # print(self.filename)

    def __getitem__(self, index):

        image_A = None
        if(self.mode=='train'):
            fullname = self.files_A[index % len(self.files_A)]
            image_A = Image.open(fullname).convert('RGB')
            sequence = int(((fullname.split("/A/")[1]).split('.')[0]).split('_seq')[1])
            sequence1 = sequence + 1
            sequence2 = sequence + 2
            nextframe = fullname.split("_seq")[0] + "_seq" + str(sequence1) + '.png'
            secondframe = fullname.split("_seq")[0] + "_seq" + str(sequence2) + '.png'
            if exists(nextframe) and exists(secondframe):
                pass
            else:
                sequence1 = sequence - 1 
                secondframe = fullname.split("_seq")[0] + "_seq" + str(sequence1) + '.png'
                sequence2 = sequence - 2 
                nextframe = fullname.split("_seq")[0] + "_seq" + str(sequence2) + '.png' 
                image_A =  Image.open(secondframe).convert('RGB')         
            image_A1 = Image.open(nextframe).convert('RGB')

            
        image_B = None
        image_B1 = None
        image_B2 = None
        if(self.mode=='train'):
            fullname = self.files_B[index % len(self.files_B)]
            image_B = Image.open(fullname).convert('RGB')
            sequence = int(((fullname.split("/B/")[1]).split('.')[0]).split('_seq')[1])
            sequence1 = sequence + 1
            sequence2 = sequence + 2
            nextframe = fullname.split("_seq")[0] + "_seq" + str(sequence1) + '.png'
            secondframe = fullname.split("_seq")[0] + "_seq" + str(sequence2) + '.png'
            if exists(nextframe) and exists(secondframe):
                image_B1 =  Image.open(nextframe).convert('RGB')
                image_B2 =  Image.open(secondframe).convert('RGB')
            else:
                sequence1 = sequence - 1 
                secondframe = fullname.split("_seq")[0] + "_seq" + str(sequence1) + '.png'
                sequence2 = sequence - 2 
                nextframe = fullname.split("_seq")[0] + "_seq" + str(sequence2) + '.png'
                image_B2 = image_B 
                image_B1 = Image.open(nextframe).convert('RGB')
                image_B =  Image.open(secondframe).convert('RGB')


        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        
        if(self.mode=='train'):
            item_A1 = self.transform(image_A1)
            item_B1 = self.transform(image_B1)
            item_B2 = self.transform(image_B2)

        if(self.mode=='train'):
            w_offset = random.randint(0, max(0, 286 - 256 - 1))
            h_offset = random.randint(0, max(0, 286 - 256 - 1))
            item_A = item_A [:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            item_B  =  item_B [:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            item_A1 = item_A1 [:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            item_B1  =  item_B1 [:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            item_B2  =  item_B2 [:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        item_A  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(item_A)
        item_B  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(item_B)
        if(self.mode=='train'):
            item_A1  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(item_A1)
            item_B1  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(item_B1)
            item_B2  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(item_B2)

        if (self.mode=='train'):
            # item_C = self.transform(Image.open(self.files_C[index % len(self.files_C)]))
            return {'A': item_A, 'B': item_B, 'A1': item_A1, 'B1': item_B1, 'B2': item_B2}
        else:
            # return  {'A': item_A, 'B': item_B, 'filename': int(self.filename[index])}
            return  {'A': item_A, 'B': item_B, 'filename': self.filename[index]}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
