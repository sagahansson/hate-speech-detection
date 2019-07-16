import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


from module import test

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import time

from tensorboardX import SummaryWriter

from tqdm import tqdm


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
            nn.ReLU(),
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
            text_features = self.text_feat_model(text)
            text_features = text_features[1]
            # print(text_features)
            features = torch.cat((features, text_features), dim=1)

        if self.USE_HATE_WORDS == 1:
            features = torch.cat((features, hate_words), dim=1)

        features = features[:, 1:]

        out = self.classifier(features)

        return out

"Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"
def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    # print(output)
    pred = output >= 0.5
    # print(pred)
    truth = target >= 0.5
    # print(truth)
    acc = pred.eq(truth).sum()
    # print(acc)
    return acc

"Credit to https://gist.github.com/kyamagu/73ab34cbe12f3db807a314019062ad43"
def validAccuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    # print(output)
    pred = output >= 0.5
    # print(pred)
    truth = target >= 0.5
    # print(truth)
    acc = pred.eq(truth)
    # print(acc)
    return acc

TOP_SIZE = 20
top_losses = []
fewer_losses = []

def getLossFromTuple(item):
    return item[1]

def validate(dataloader_valid, criterion, device):
    # t = time.time()

    loss = 0
    acc = 0
    i = 0
    global top_losses
    global fewer_losses

    top_losses = []
    fewer_losses = []

    for batch in dataloader_valid:

        # print("loadtime, ", time.time() - t)

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

            for j, x in enumerate(distances):
                top_losses.append([paths_batch[j], x])
                fewer_losses.append([paths_batch[j], x])

            top_losses.sort(key=getLossFromTuple, reverse=True)
            fewer_losses.sort(key=getLossFromTuple, reverse=False)

            top_losses = top_losses[:TOP_SIZE]
            fewer_losses = fewer_losses[:TOP_SIZE]

            acc += kk.sum()


            size = target_batch.numel()
            loss += criterion(pred, target_batch) * size

            i += target_batch.numel()


    valid_acc = acc.float()/i
    valid_mse = loss/i
    print('acc', acc)
    print('i', i)
    # t = time.time()
    return valid_acc, valid_mse


if __name__ == '__main__':

    HIDDEN_SIZE = 50
    N_EPOCHS = 300
    BATCH_SIZE = 30

    UNFREEZE_FEATURES = 10

    USE_IMAGE = 1
    USE_TEXT = 0
    USE_HATE_WORDS = 0

    TRAIN_METADATA_HATE = "hateMemesList.txt.train"
    TRAIN_METADATA_GOOD = "redditMemesList.txt.train"
    VALID_METADATA_HATE = "hateMemesList.txt.valid"
    VALID_METADATA_GOOD = "redditMemesList.txt.valid"
    BASE_PATH = "data/train_data"
    # MODEL_SAVE = "models/matching.pt"
    # MODEL_SAVE = "models/matching_pretrained.pt"
    MODEL_SAVE = "models/classifier.pt"

    # logname = "H100x4_IF1000v2"
    # logname = "H100x4_matching_pretrained"
    # logname = "logs_H50/H50x4_Dropoutv2"
    # logname = "logs_H50/unfreeze_bert_textonly"
    logname = "kk2"

    # checkpoint = "models/unsupervised_pretrain.pt"
    checkpoint = None


    start_time = time.time()
    writer = SummaryWriter("logs/" + logname)

    # Configuring CUDA / CPU execution
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('device', device)


    hate_list = [
            'ali baba',
            'allah',
            'abbo',
            # 'ape',
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
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case="true")
    # Get Textual Embedding.
    bert_model = BertModel.from_pretrai ned("bert-base-multilingual-cased")
    bert_model.eval()
    bert_model.to(device)

    # VGG16 output --> 1000 features
    # VGG16 -1 layer --> 4096 features
    # BERT text embedder --> 768 features

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
        full_model.load_state_dict(torch.load(checkpoint))

    full_model.to(device)
    # print(full_model)

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

    train_dataset = test.ImagesDataLoader(TRAIN_METADATA_GOOD, TRAIN_METADATA_HATE, BASE_PATH, transform)
    valid_dataset = test.ImagesDataLoader(VALID_METADATA_GOOD, VALID_METADATA_HATE, BASE_PATH, transformValid)
    # train_dataset = test.ImageTextMatcherDataLoader(TRAIN_METADATA_GOOD, TRAIN_METADATA_HATE, BASE_PATH, transform)
    # valid_dataset = test.ImageTextMatcherDataLoader(VALID_METADATA_GOOD, VALID_METADATA_HATE, BASE_PATH, transformValid)

    DATASET_LEN = train_dataset.__len__()

    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)
    dataloader_valid = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test.custom_collate)


    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    # feature_parameters = list(full_model.im_feat_model.classifier.parameters()) + list(bert_model.parameters())
    #
    # optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9)
    # optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(full_model.classifier.parameters())
    # features_optimizer = torch.optim.Adam(feature_parameters)
    features_optimizer = torch.optim.Adam(VGG16_features.classifier.parameters(), lr=0.00005)


    iteration = 0

    best_acc = -1

    for i in range(N_EPOCHS):

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

            # forward_init = time.time()
            pred = full_model(image_batch, text_batch, hate_words_batch)
            # forward_final = time.time()
            # print("forward time: ", forward_final - forward_init)

            loss = criterion(pred, target_batch)

            loss.backward()


            optimizer.step()

            if i >= UNFREEZE_FEATURES:
                features_optimizer.step()

            writer.add_scalar('train/mse', loss, iteration*BATCH_SIZE)
            iteration += 1

            pbar.update(BATCH_SIZE)
            #
            # break
            #
            # print(pred)
            # pass

        # print("total hate hate, ", HateWordsVector.total_hate_hate)
        # print("total hate love, ", HateWordsVector.total_hate_love)
        # print("total hate, ", HateWordsVector.total_hate_hate + HateWordsVector.total_hate_love)
        # print("total sentences hate", HateWordsVector.total_sentences_hate)
        # print("empty text", HateWordsVector.empty_text)
        # quit()


        epoch_end = time.time()

        print("Epoch time elapsed:", epoch_end - epoch_init)

        print("Saving full model to " + MODEL_SAVE)
        torch.save(full_model.state_dict(), MODEL_SAVE)
        print("Starting Validation")

        valid_init = time.time()

        full_model.eval()
        valid_acc, valid_loss = validate(dataloader_valid, criterion, device)

        full_model.train()

        if valid_acc > best_acc:
            torch.save(full_model.state_dict(), MODEL_SAVE + '.best')
            best_acc = valid_acc

            accs = open("results/accuracys", "w")

            accs.write("Smaller losses from best acc (epoch : " + str(i) + ")\n")
            #
            # global fewer_losses
            # global top_losses
            #
            # print(fewer_losses)
            # print(top_losses)

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
        # print("Validation accuracy on epoch " + str(i) + ": ")
        # break


    end_time = time.time()
    print("Elapsed Time:", end_time - start_time)


    accs.close()

    writer.close()
