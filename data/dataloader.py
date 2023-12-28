import glob
import itertools
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as tf
import os

random.seed(2024)

################################################# R to S训练数据加载器 ##################################################

class TrainDataloader_RS(Dataset):

    def __init__(self, style_path, content_path, transform=None,unaligned=True, model="train"):

        self.transform = tf.Compose(transform)
        self.unaligned = unaligned
        self.model = model

        self.style_path = os.path.join(style_path,"*.*")
        self.content_path = os.path.join(content_path, "*.*")

        self.list_style = sorted(glob.glob(self.style_path))
        self.list_content = sorted(glob.glob(self.content_path))

        print("Total {} examples:".format(model), max(len(self.list_style), len(self.list_content)))


    def __getitem__(self, index):

        if self.unaligned:

            style = self.list_style[random.randint(0, len(self.list_style) - 1)]

        else:

            style = self.list_style[index % len(self.list_style)]

        content = self.list_content[index % len(self.list_content)]
        name = os.path.basename(content)

        style = Image.open(style).convert("RGB")
        content = Image.open(content).convert("RGB")

        style = self.transform(style)
        content = self.transform(content)

        return style,content,name

    def __len__(self):

        return max(len(self.list_style),len(self.list_content))


################################################# S to R训练数据加载器 ##################################################

def Randomized_Style_Version(stylized_path):
    '''
    :return: image from Style Bank
    '''
    random_list = random.sample(range(5), 5)
    hazy_list = []
    style_list = ["stylized150/","stylized160/","stylized170/","stylized180/","stylized190/"]
    for i in range(len(random_list)):
        random_index =  random_list[i]
        random_path = stylized_path + style_list[random_index]
        random_path_list = os.path.join(random_path, "*.*")

        hazy_list.append(random_path_list)

    return hazy_list


class TrainDataloader_SR(Dataset):

    def __init__(self, hazy_path, clear_path, stylized_path,transform=None, model="train"):

        self.transform = tf.Compose(transform)
        self.model = model
        self.list_random_hazy = []

        self.hazy_path = os.path.join(hazy_path,"*.*")
        self.clear_path = os.path.join(clear_path, "*.*")
        self.stylized_path = stylized_path

        self.list_hazy = sorted(glob.glob(self.hazy_path))
        self.list_clear = sorted(glob.glob(self.clear_path))

        self.random_style = Randomized_Style_Version(self.stylized_path)

        for i in range(len(self.random_style)):

            self.list_random_hazy.append(sorted(glob.glob(self.random_style[i])))

        print("Total {} examples:".format(model), max(len(self.list_hazy), len(self.list_clear)))


    def __getitem__(self, index):

        hazy = self.list_hazy[index % len(self.list_hazy)]
        clear = self.list_clear[index % len(self.list_clear)]

        name = os.path.basename(hazy)

        hazy = Image.open(hazy).convert("RGB")
        clear = Image.open(clear).convert("RGB")

        Candidate1 = self.list_random_hazy[0][random.randint(0, len(self.list_random_hazy[0]) - 1)]
        Candidate2 = self.list_random_hazy[1][random.randint(0, len(self.list_random_hazy[1]) - 1)]
        Candidate3 = self.list_random_hazy[2][random.randint(0, len(self.list_random_hazy[2]) - 1)]
        Candidate4 = self.list_random_hazy[3][random.randint(0, len(self.list_random_hazy[3]) - 1)]
        Candidate5 = self.list_random_hazy[4][random.randint(0, len(self.list_random_hazy[4]) - 1)]

        # print(Candidate1,Candidate2,Candidate3,Candidate4,Candidate5)

        hazy = self.transform(hazy)
        clear = self.transform(clear)

        Candidate1 = self.transform(Image.open(Candidate1).convert("RGB"))
        Candidate2 = self.transform(Image.open(Candidate2).convert("RGB"))
        Candidate3 = self.transform(Image.open(Candidate3).convert("RGB"))
        Candidate4 = self.transform(Image.open(Candidate4).convert("RGB"))
        Candidate5 = self.transform(Image.open(Candidate5).convert("RGB"))

        Candidate_list = [Candidate1,Candidate2,Candidate3,Candidate4,Candidate5]

        return hazy,clear,name,Candidate_list

    def __len__(self):

        return max(len(self.list_hazy),len(self.list_clear))



class TestDataloader(Dataset):

    def __init__(self, haze_path, clear_path, transform=None, model="test"):

        self.transform = tf.Compose(transform)
        self.model = model

        self.haze_path = os.path.join(haze_path,"*.*")
        self.clear_path = os.path.join(clear_path,"*.*")

        self.list_haze = sorted(glob.glob(self.haze_path))
        self.list_clear = sorted(glob.glob(self.clear_path))

        print("Total {} examples:".format(model), max(len(self.list_haze), len(self.list_clear)))


    def __getitem__(self, index):

        haze = self.list_haze[index % len(self.list_haze)]

        clear = self.list_clear[index % len(self.list_clear)]

        name = os.path.basename(haze)

        haze = Image.open(haze).convert("RGB")
        clear = Image.open(clear).convert("RGB")

        w = haze.size[0]
        h = haze.size[1]
        size = [h,w]

        haze = self.transform(haze)
        clear = self.transform(clear)

        return haze, clear, name,size

    def __len__(self):

        return max(len(self.list_haze),len(self.list_clear))




if  __name__ == "__main__":
     haze_path= "../results/stylized150/"
     clear_path= "../results/stylized150/"
     style_path= "../results/"
     transform_ = [tf.ToTensor()]

     train_sets=TrainDataloader_SR(haze_path,clear_path,style_path,transform_)

     dataload = DataLoader(train_sets,batch_size=1,shuffle=True,num_workers=4)

     for i, batch in enumerate(dataload):

         print((batch[8].shape))