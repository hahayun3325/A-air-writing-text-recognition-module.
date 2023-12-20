import os
import sys

sys.path.append('./tr')
import string
import argparse
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms

from tr.dataset import AlignCollate, RawDataset
from tr.model import Model
from tr.utils import CTCLabelConverter, AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

def tr_recognize(opt, model):
    # word = demo(opt, model)
    word = getText(opt, model)
    if len(word):
        return word
    else:
        return None


def getParser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--image_folder', default='./tr/saved_images',
                        help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model',
                        default='./tr/saved_models/TPS-MobileNetV2-BiLSTM-Attn-Seed1111_new/best_accuracy.pth',
                        help="path to saved_model to evaluation")
    # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--character', type=str, default='abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')

    """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='MobileNetV2',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet|Ghost|MobileNetV2|MobileNetV3')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    return opt


def loadModel(opt):
    model = Model(opt)

    # print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #       opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #       opt.SequenceModeling, opt.Prediction)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    return model


def demo(opt, model):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            # log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":30s}\t{"predicted_labels":25s}\tconfidence score'
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                print(f'{img_name:30s}\t{pred:25s}\t{confidence_score:0.4f}')
                # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            # log.close()
    return pred


def getText(opt, model):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)

    path = opt.image_folder + '/character.png'
    # img = Image.open(path).convert('L')
    extractor = myNormalize()
    img = extractor(path)
    pred, _ = infer_origin(model, img, opt, converter)

    return pred


# 读取图片并调整大小归一化
class myNormalize:
    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        if type(img) == str:
            img = Image.open(img).convert('L')
        img = img.resize((self.imgW, self.imgH), Image.BICUBIC)
        img = self.toTensor(img)
        img = torch.unsqueeze(img, dim=0)
        return img


# 根据手写轨迹图片识别出单词和置信度
def infer_origin(model, img, opt, converter):
    model.eval()
    with torch.no_grad():

        image = img.to(device)

        length_for_pred = torch.IntTensor([opt.batch_max_length] * 1).to(device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)  # 贪婪取每一个概率的最大值
        preds_str = converter.decode(preds_index, length_for_pred)  # 解码，索引转文本

        preds_prob = F.softmax(preds, dim=2)  # 概率归一化
        preds_max_prob, _ = preds_prob.max(dim=2)  # 取出最大概率

        pred_EOS = preds_str[0].find('[s]')  # 找到结尾位置
        pred = preds_str[0][:pred_EOS]  # prune after "end of sentence" token ([s])
        pred_max_prob = preds_max_prob[0][:pred_EOS]  # 所有字符概率列表

        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

    return pred, float(confidence_score)
