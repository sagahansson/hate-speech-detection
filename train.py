import sys
sys.path.append("../..")
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from module import test
from pytorch_transformers import *
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

import json
import argparse
torch.cuda.empty_cache()

import os
if "models" not in os.listdir("./"):
    os.mkdir("./models/") 

class MultimodalClassifier(nn.Module):
    def __init__(self, image_feat_model, text_feat_model, TOTAL_FEATURES,
                 USE_IMAGE, USE_TEXT, USE_HATE_WORDS, hidden_size, device):

        super(MultimodalClassifier, self).__init__()
        self.im_feat_model = image_feat_model
        self.text_feat_model = text_feat_model

        self.USE_IMAGE = USE_IMAGE
        self.USE_TEXT = USE_TEXT
        self.USE_HATE_WORDS = USE_HATE_WORDS

        self.TOTAL_FEATURES = TOTAL_FEATURES

        self.classifier = nn.Sequential(
            nn.Linear(TOTAL_FEATURES, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), #original
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, 1),
            # nn.Softmax()
        )

        self.device = device


    def forward(self, image, text, hate_words):

        # with torch.no_grad():
        if self.USE_IMAGE == 1:
            batch_size = image.size()[0]
        elif self.USE_TEXT == 1:
            batch_size = image.size()[0]
        elif self.USE_HATE_WORDS == 1:
            batch_size = image.size()[0]

        features = torch.zeros(batch_size, 1).to(device)

        if self.USE_IMAGE == 1:
            image_features = self.im_feat_model(image)
            features = torch.cat((features, image_features), dim=1)

        if self.USE_TEXT == 1:
            last_hidden_states = self.text_feat_model(text)[0]
            text_features = torch.sum(last_hidden_states, dim=1)
            text_features = text_features / last_hidden_states.size()[1]
            features = torch.cat((features, text_features), dim=1)

        if self.USE_HATE_WORDS == 1:
            features = torch.cat((features, hate_words), dim=1)

        features = features[:, 1:]

        out = self.classifier(features)

        return out

"Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"
def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum()
    return acc

"Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"
def validAccuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth)
    return acc

TOP_SIZE = 20
top_losses = []
fewer_losses = []

def getLossFromTuple(item):
    return item[1]

def validate(dataloader_valid, criterion, device):

    loss = 0
    acc = 0
    i = 0
    global top_losses
    global fewer_losses

    top_losses = []
    fewer_losses = []

    for batch in dataloader_valid:

        image_batch = batch["image"].to(device)
        text_batch = batch["bert_tokens"].to(device)
        hate_words_batch = batch["hate_words"].to(device)
        paths_batch = batch["image_paths"]

        target_batch = batch["class"].to(device)
        target_batch = target_batch.unsqueeze(1)



        with torch.no_grad():

            pred = full_model(image_batch, text_batch, hate_words_batch)

            distances = (pred - target_batch) ** 2

            kk = validAccuracy(pred, target_batch)
            acc += kk.sum()

            for j, x in enumerate(distances):
                top_losses.append([paths_batch[j], x])
                fewer_losses.append([paths_batch[j], x])

            top_losses.sort(key=getLossFromTuple, reverse=True)
            fewer_losses.sort(key=getLossFromTuple, reverse=False)

            top_losses = top_losses[:TOP_SIZE]
            fewer_losses = fewer_losses[:TOP_SIZE]

            size = target_batch.numel()
            loss += criterion(pred, target_batch) * size

            i += target_batch.numel()


    valid_acc = acc.float()/i
    valid_mse = loss/i
    #print('acc', acc)
    #print('i', i)
    return valid_acc, valid_mse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Input name and hyperparameters of the model.")
        
    parser.add_argument("name", type=float, 
                        help="The base name of the classifier (int).")
    
    parser.add_argument("--hidsize", "-HS", type=int, default=20,
                        help="Size of hidden layers. Default: 20.")
    
    parser.add_argument("--epochs", "-E", type=int, default=40,
                        help="Number of epochs to train for. Default: 40.")
    
    parser.add_argument("--batchsize", "-BS", type=int, default=50, 
                        help="Size of the batches. Default: 50.")
    
    parser.add_argument("--balanced", "-B", dest="balanced", type=int, default=1, choices=[0, 1], 
                        help="Whether or not to use balanced data; 0 = no, 1 = yes. Default: 1.")
    
    parser.add_argument("--test", "-T", dest="tst", type=int, default=0, choices=[0, 1],  
                        help="Whether or not to test; 0 = No, 1 = Yes. Default: 0.")
    
    parser.add_argument("--img", dest="img", type=int, default=1, choices=[0, 1],  
                        help="Whether or not to use image as feature; 0 = No, 1 = Yes. Default: 1.")
    
    parser.add_argument("--txt", dest="txt", type=int, default=1, choices=[0, 1],  
                        help="Whether or not to use text as feature; 0 = No, 1 = Yes. Default: 1.")
    
    parser.add_argument("--lr", dest="lr", type=float, default=0.001,  
                        help="Rate at which the model learns. Default: 0.001.")

    parser.add_argument("--cpt", dest="cpt", type=bool, default=False,  
                        help="Whether or not to start from checkpoint. Default: False.")
    
    parser.add_argument("--dev", dest="device", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Which GPU to train on. Default: 1.")
    
    
    args = parser.parse_args()
    
    
    TEST_BIN = args.tst
    HIDDEN_SIZE = args.hidsize
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    print("test: ", TEST_BIN)
    
    BASE_PATH = "../../data/data"
    nr = args.name

    BALANCED = args.balanced
    
    USE_IMAGE = args.img
    USE_TEXT = args.img
    USE_HATE_WORDS = 0
    LR = args.lr
    
    MODEL_SAVE = "models/clf" + str(nr) + "." + str(USE_IMAGE) + str(USE_TEXT) + ".pt" if BALANCED == 0 else "models/clf" + str(nr) + ".b" + str(USE_IMAGE) + str(USE_TEXT) + ".pt"
    logname = "logs_new/" + MODEL_SAVE.split("/")[1]
    
    print(f"Model save location: {MODEL_SAVE}")
    
    UNFREEZE_FEATURES = 999
            
    checkpoint = None    
    if args.cpt:
        checkpoint = MODEL_SAVE + ".best"
        print("Loading from checkpoint.")

    hp_dict = {"hid size"   : str(HIDDEN_SIZE),
               "n epochs"   : str(N_EPOCHS),
               "batch size" : str(BATCH_SIZE),
               "image"      : str(USE_IMAGE),
               "text"       : str(USE_TEXT),
               "lr"         : str(LR)}
    
    logfiles = [os.path.join(top, file) for top, dirs, files in os.walk("models") for file in files]
   
    if (MODEL_SAVE + ".hp.json") not in logfiles:
        with open(MODEL_SAVE + ".hp.json", "w") as hp:
            json.dump(hp_dict, hp) 
            print("Saved hyperparameters.")
    else:
        with open(MODEL_SAVE + ".hp.json", "r") as hps:
            hp_dict = json.load(hps)

            HIDDEN_SIZE = int(hp_dict["hid size"])
            N_EPOCHS    = int(hp_dict["n epochs"])
            BATCH_SIZE  = int(hp_dict["batch size"])
            USE_IMAGE   = int(hp_dict["image"])
            USE_TEXT    = int(hp_dict["text"])
            LR          = float(hp_dict["lr"])

            print("Loaded hyperparameters")

    
    TRAIN_METADATA_HATE = "hateMemesList.txt.train"
    #TRAIN_METADATA_GOOD = "redditMemesList.txt.train" ## from original code
    TRAIN_METADATA_GOOD = "goodMemesList.txt.train"
    
    VALID_METADATA_HATE = "hateMemesList.txt.val"
    #VALID_METADATA_GOOD = "redditMemesList.txt.valid" 
    VALID_METADATA_GOOD = "goodMemesList.txt.val"
    
    TEST_METADATA_HATE = "hateMemesList.txt.test"
    TEST_METADATA_GOOD = "goodMemesList.txt.test"
    
    META = [TRAIN_METADATA_HATE, TRAIN_METADATA_GOOD, VALID_METADATA_HATE, VALID_METADATA_GOOD, TEST_METADATA_HATE, TEST_METADATA_GOOD]
    
    if BALANCED:
        META = ["bal." + x for x in META]
        TRAIN_METADATA_HATE = META[0]
        TRAIN_METADATA_GOOD = META[1] 
        VALID_METADATA_HATE = META[2]
        VALID_METADATA_GOOD = META[3]
        TEST_METADATA_HATE  = META[4]
        TEST_METADATA_GOOD  = META[5]
        print("Got balanced data")
        
    start_time = time.time()
    writer = SummaryWriter("logs/" + logname)
    # Configuring CUDA / CPU execution
    device = torch.device(("cuda:" + str(args.device)) if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # Keywords (deprecated)
    hate_list = [
            'ali baba',
            'allah',
            'abbo',
            'black',
            'bomb',
            'dynamite',
            'jew',
            'nazi',
            'niglet',
            'nigger',
            'nigga',
            'paki',
        ]


    # Get image descriptor
    VGG16_features = torchvision.models.vgg16(pretrained=True)
    VGG16_features.classifier = VGG16_features.classifier[:-3]

    VGG16_features.to(device)

    # To embed text, we use a Pytorch implementation of BERT: Using pythorch BERT implementation from https://github.com/huggingface/pytorch-pretrained-BERT
    # Get Textual Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    # Get Textual Embedding.
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    bert_model.eval()
    bert_model.to(device)
    
    print(f"Got VGG \nGot BERT")
    # IMAGE_AND_TEXT_FEATURES = 1768
    IMAGE_FEATURES = 4096
    TEXT_FEATURES = 768
    HATE_WORDS = len(hate_list)

    IMAGE_AND_TEXT_FEATURES = IMAGE_FEATURES * USE_IMAGE + TEXT_FEATURES * USE_TEXT\
                              + HATE_WORDS * USE_HATE_WORDS

    full_model = MultimodalClassifier(VGG16_features, bert_model,
                                      IMAGE_AND_TEXT_FEATURES, USE_IMAGE,
                                      USE_TEXT, USE_HATE_WORDS, HIDDEN_SIZE,
                                      device)

    if checkpoint is not None:
        cp = torch.load(checkpoint)
        full_model.load_state_dict(cp)
        
        
    full_model.to(device)

    # transform = transforms.Compose([test.Rescale((256, 256)),
    transform = transforms.Compose([test.Rescale((224, 224)),
                                    # test.RandomCrop(224),
                                    test.HateWordsVector(hate_list),
                                    test.Tokenize(tokenizer),
                                    test.ToTensor()])

    transformValid = transforms.Compose([test.Rescale((224, 224)),
                                    # test.RandomCrop(224),
                                    test.HateWordsVector(hate_list),
                                    test.Tokenize(tokenizer),
                                    test.ToTensor()])
    
    transformTest = transforms.Compose([test.Rescale((224, 224)),
                                    # test.RandomCrop(224),
                                    test.HateWordsVector(hate_list),
                                    test.Tokenize(tokenizer),
                                    test.ToTensor()])

    train_dataset = test.ImagesDataLoader(TRAIN_METADATA_GOOD, TRAIN_METADATA_HATE, BASE_PATH, transform)
    valid_dataset = test.ImagesDataLoader(VALID_METADATA_GOOD, VALID_METADATA_HATE, BASE_PATH, transformValid)
    test_dataset = test.ImagesDataLoader(TEST_METADATA_GOOD, TEST_METADATA_HATE, BASE_PATH, transformTest)
    # train_dataset = test.ImageTextMatcherDataLoader(TRAIN_METADATA_GOOD, TRAIN_METADATA_HATE, BASE_PATH, transform)
    # valid_dataset = test.ImageTextMatcherDataLoader(VALID_METADATA_GOOD, VALID_METADATA_HATE, BASE_PATH, transformValid)

    DATASET_LEN = train_dataset.__len__()

    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)
    dataloader_valid = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)
    dataloader_test  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)
    print("GOT DATA THANKS")
        # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    
    if TEST_BIN:
        print("LET'S GO TEST BRO")
        full_model.load_state_dict(torch.load(checkpoint))
        full_model.to(device)
        full_model.eval()
        full_model.text_feat_model.eval()
        full_model.im_feat_model.eval()
        test_acc, test_loss = validate(dataloader_test, criterion, device)
        with open(MODEL_SAVE + ".testacc.log", "w+") as accLog:
            accLog.write(f"Test acc: {test_acc}, \nTest_loss: {test_loss}")
        print(f"Test acc: {test_acc}, \nTest_loss: {test_loss}")
        alskdlaksd
    
    print("Got through dataloaders, start training!")

    # feature_parameters = list(full_model.im_feat_model.classifier.parameters()) + list(bert_model.parameters())
    #
    # optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    # optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(full_model.classifier.parameters(), lr=LR)
    # features_optimizer = torch.optim.Adam(feature_parameters)
    features_optimizer = torch.optim.Adam(VGG16_features.classifier.parameters())


    iteration = 0

    best_acc = np.array(-1.00)
    full_model.text_feat_model.train()
    full_model.im_feat_model.train()

    for i in range(N_EPOCHS):
        print("Epoch:", i)
        epoch_init = time.time()
        pbar = tqdm(total=DATASET_LEN)
        for batch in dataloader_train:
            image_batch = batch["image"].to(device)
            text_batch = batch["bert_tokens"].to(device)
            hate_words_batch = batch["hate_words"].to(device)

            target_batch = batch["class"].to(device)

            target_batch = target_batch.unsqueeze(1)

            optimizer.zero_grad()
            features_optimizer.zero_grad()

            pred = full_model(image_batch, text_batch, hate_words_batch)

            loss = criterion(pred, target_batch)

            loss.backward()


            optimizer.step()

            if i >= UNFREEZE_FEATURES:
                features_optimizer.step()
            with open(MODEL_SAVE + ".accloss.log", "a+") as accloss:
                accloss.write("val loss:" + str(loss) + "; " + "epoch: " + str(i) + "\n")

            writer.add_scalar('train/mse', loss, iteration*BATCH_SIZE)
            
            iteration += 1

            pbar.update(BATCH_SIZE)


        epoch_end = time.time()

        print("Epoch time elapsed:", epoch_end - epoch_init)

        print("Starting Validation!")

        valid_init = time.time()

        full_model.eval()
        full_model.text_feat_model.eval()
        full_model.im_feat_model.eval()
        valid_acc, valid_loss = validate(dataloader_valid, criterion, device)
        full_model.text_feat_model.train()
        full_model.im_feat_model.train()
        full_model.train()
        
        with open(MODEL_SAVE + ".valloss.log", "a+") as vallog:
            vallog.write("val loss:" + str(valid_loss) + "; " + "epoch: " + str(i) + "\n")
        
        print(f"Validation loss: {valid_loss}")
        print(f"Validation accuracy: {valid_acc}")
        valid_acc_np = valid_acc.cpu().numpy()
        #print(type(best_acc))
        torch.save(full_model.state_dict(), "./" + MODEL_SAVE)
        if type(best_acc) != np.ndarray:
            best_acc = best_acc.cpu().numpy()
        #print(type(best_acc))
        if valid_acc_np > best_acc:
            #best_epoch = i
            print("Saving full model to " + "./" + MODEL_SAVE + ".best")
            torch.save(full_model.state_dict(), "./" + MODEL_SAVE + '.best')
            logfile = open(MODEL_SAVE + ".best.log", "a+")
            best_acc = valid_acc_np
            logfile.write("best_acc:" + str(valid_acc_np)+"\n" + "best epoch: " + str(i) + "\n")
            logfile.close()
            best_acc = valid_acc
#        else: # honestly not sure what ive done here but im leaving it until i remember 
#            diff = i - best_epoch
#            if diff > 5: 
                

            accs = open("results/accuracys", "w")

            accs.write("Smaller losses from best acc (epoch : " + str(i) + ")\n")

            for x in fewer_losses:

                accs.write(str(x[0]) + "\t" + str(x[1]) + "\n")

            accs.write("Top Loss:\n")
            for x in top_losses:
                accs.write(str(x[0]) + "\t" + str(x[1]) + "\n")

            accs.close()

        valid_end = time.time()
        print("Time in validation:", valid_end - valid_init)
        writer.add_scalar('validation/valid_accuracy', valid_acc, i+1)
        writer.add_scalar('validation/valid_mse', valid_loss, i+1)

    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)

    accs.close()

    writer.close()
