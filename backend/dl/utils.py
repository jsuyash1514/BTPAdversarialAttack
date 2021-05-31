import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import mnist
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import advertorch
from PIL import Image
from torchvision.transforms import ToTensor
import pickle
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
import json
from torchvision.utils import save_image
import time


class detect_AE(nn.Module):
    def __init__(self, lr=0.1):
        super(detect_AE, self).__init__()
        self.encoder1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)
        self.encoder2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)

        self.decoder1 = nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.decoder2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)

        self.out = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)

    def forward(self, input):
        enc1 = F.max_pool2d(F.relu(self.encoder1(input)), kernel_size=(2,2), stride=2)
        code = F.max_pool2d(F.relu(self.encoder2(enc1)), kernel_size=(2,2), stride=1)

        dec1 = F.relu(self.decoder1(code))
        dec2 = F.relu(self.decoder2(dec1))


        out = torch.sigmoid(self.out(dec2))

        return out, code
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784,256)
        self.relu = nn.ReLU(True)
        self.lin2 = nn.Linear(256,64)
        self.lin3 = nn.Linear(64,256)
        self.lin4 = nn.Linear(256,784)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.sig(self.lin4(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

detect_model = detect_AE()
denoise_model = Autoencoder()
strong_adv_model = Model()
strong_base_model = Model()
black_box_model = Model()

detect_model.load_state_dict(torch.load('./dl/detect_weights.pkl'))
denoise_model.load_state_dict(torch.load('./dl/denoise_weights.pkl'))
black_box_model.load_state_dict(torch.load('./dl/BlackBox.pkl'))
strong_adv_model.load_state_dict(torch.load('./dl/Strong_adver.pkl'))
strong_base_model.load_state_dict(torch.load('./dl/Strong_base.pkl'))


gm = GaussianMixture(n_components=10)
with open('./dl/model_GMM.pkl', 'rb') as file:
    gm = pickle.load(file)

def lenet_model_img(img_path):
  import tensorflow.keras
  import numpy as np
  from tensorflow.keras.models import model_from_json
  from PIL import Image
  from numpy import asarray

  # loading model
  # load json and create model
  json_file = open('./dl/Le_netmodel.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("./dl/le_netmodel.h5")
  # print("Loaded model from disk")

  img = Image.open(img_path)
  numpydata = asarray(img)
  prvdimg = numpydata/255.0
  prvdimg2=prvdimg.reshape(1,28,28,1)
  resclass=loaded_model.predict_classes(prvdimg2)
  res=loaded_model.predict(prvdimg2)
  totalsum=0.0
  prdprob=[]
  for i in res[0]:
    totalsum+=i


  for i in res[0]:
    prdprob.append(i/totalsum)

  data = {"lenet_class":int(resclass[0]), "lenet_accuracy":max(prdprob)}
  return json.dumps(data)

def lenet_model_np(np_array):
  import tensorflow.keras
  import numpy as np
  from tensorflow.keras.models import model_from_json
  from PIL import Image
  from numpy import asarray

  # loading model
  # load json and create model
  json_file = open('./dl/Le_netmodel.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("./dl/le_netmodel.h5")
  # print("Loaded model from disk")

  numpydata = np_array
  prvdimg = numpydata/255.0
  prvdimg2=prvdimg.reshape(1,28,28,1)
  resclass=loaded_model.predict_classes(prvdimg2)
  res=loaded_model.predict(prvdimg2)
  totalsum=0.0
  prdprob=[]
  for i in res[0]:
    totalsum+=i


  for i in res[0]:
    prdprob.append(i/totalsum)

  data = {"lenet_class":int(resclass[0]), "lenet_accuracy":max(prdprob)}
  return json.dumps(data)


def f(input_image_name):
    

    #code for detection and classification of normal image
    img_path= input_image_name
    normal_lenet = lenet_model_img(img_path)

    # lenet_data=lenet_model(img_path)
    # return lenet_data

    image = Image.open(img_path)

    image = ToTensor()(image).unsqueeze(0) 
    image = Variable(image)
    
    detect_model.eval()
    black_box_model.eval()
    alpha_criterion = nn.MSELoss()
    strong_adv_model.eval()
    strong_base_model.eval()
    denoise_model.eval()

    thresh = 0.065
    lamda = 0.0
    
    adversary = advertorch.attacks.LinfPGDAttack(black_box_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3, nb_iter=40, eps_iter=0.02,    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    adv_untargeted = adversary.perturb(image)
    
    pth = "./media/results/advimg{}.jpg".format(round(time.time() * 1000))
    img_adver = adv_untargeted[0]
    img_adver_np = img_adver.numpy()
    adver_lenet = lenet_model_np(img_adver_np)
    save_image(img_adver,pth)
    with torch.no_grad():
    
        out, enc = detect_model(image.float())
        alpha = alpha_criterion(out,image)
        enc = enc.reshape(32)
        beta= gm.score([enc.numpy()])

        if alpha.item() + lamda*(1/beta) < thresh:
            #print("detected true")
            str_out = "Detected image to be true sample"
            image = image.view(-1,784)
            out_denoised = denoise_model(image)
            out_denoised = out_denoised.view(-1,1,28,28)
            predict = strong_base_model(out_denoised.float())
            predict_lab = np.argmax(predict, axis=-1)
            #print("\n The predicted label for the image is: {}".format(predict_lab))
            

        if alpha.item() + lamda*(1/beta) > thresh:
            #print("detected adversarial")
            str_out = "Detected image to be adversarial sample"
            predict = strong_adv_model(image.float())
            predict_lab = np.argmax(predict, axis=-1)
            #print("\n The predicted label for the image is: {}".format(predict_lab)) 
    #code-end  

    data = {"detection_output_normal":str_out, "label_normal":predict_lab.item(), "address":pth[1:], "adv_lenet":adver_lenet, "normal_lenet":normal_lenet}   
    return json.dumps(data)


def g(input_image_name):
    
    #code for adversarial image detection and testing
    img_path= input_image_name
    image = Image.open(img_path)
    image = ToTensor()(image).unsqueeze(0) 
    image = Variable(image)

    detect_model.eval()
    black_box_model.eval()
    alpha_criterion = nn.MSELoss()
    strong_adv_model.eval()
    strong_base_model.eval()
    denoise_model.eval()

    thresh = 0.050
    lamda = 0.0


    adversary = advertorch.attacks.LinfPGDAttack(black_box_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3, nb_iter=40, eps_iter=0.02,    rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    adv_untargeted = adversary.perturb(image)

    with torch.no_grad():
        out_adver,enc_adver = detect_model(adv_untargeted.float())
        alpha = alpha_criterion(out_adver,adv_untargeted)
        enc_adver = enc_adver.reshape(32)
        beta= gm.score([enc_adver.numpy()])

        if alpha.item() + lamda*beta < thresh:
            #print("detected true")
            str_out = "Detected image true sample"
            adv_untargeted = adv_untargeted.view(-1,784)
            out_denoised = denoise_model(adv_untargeted)
            out_denoised = out_denoised.view(-1,1,28,28)
            predict = strong_base_model(out_denoised.float())
            predict_lab = np.argmax(predict, axis=-1)
            #print("\n The predicted label for the image is: {}".format(predict_lab))

        if alpha.item() + lamda*beta > thresh:
            #print("detected adversarial")
            str_out = "Detected image to be adversarial sample"
            predict = strong_adv_model(adv_untargeted.float())
            predict_lab = np.argmax(predict, axis=-1)
            #print("\n The predicted label for the image is: {}".format(predict_lab))
    
    data = {"detection_output_adver":str_out, "label_adver":predict_lab.item()}
    return json.dumps(data)
