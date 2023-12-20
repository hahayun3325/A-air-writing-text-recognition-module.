from os import path
import os
from aip import AipOcr
from PIL import Image


# 利用百度api实现图片文本识别


# 调整图片大小，对于过大的图片进行压缩
# picfile:    图片路径
# outdir：    图片输出路径
def convert_img(picfile, outdir):
    img = Image.open(picfile)
    width, height = img.size
    while width * height > 4000000:  # 该数值压缩后的图片大约 两百多k
        width = width // 2
        height = height // 2
    new_img = img.resize((width, height), Image.BILINEAR)
    new_img.save(path.join(outdir, os.path.basename(picfile)))


"""
    文字识别用的是百度通用文字识别api 
    在百度AI开放平台中，登录自己的百度账号，
    顶部大标题栏 ：开放能力 >> 文字识别 >> 通用文字识别
    可以领取免费的额度，超出后需要付费（日常测试够用了）
    选择“创建应用” 在应用列表中，能够看到自己刚刚创建好的文字识别服务了，
    将自己应用中的“AppID”，“API Key”，“Secret Key”，分别填到baiduOCR函数中的对应位置 
    使用前要注意，需先开通网络图片识别（文字识别/概览）
"""


def baidu_ocr(picfile):
    filename = path.basename(picfile)

    APP_ID = '25654109'
    API_KEY = 'M4BxtXVl2gLhiPsSmR3jvsQv'
    SECRECT_KEY = '55kHYdQx63kEnmmc94ijo0U5wfCzMHwF'
    client = AipOcr(APP_ID, API_KEY, SECRECT_KEY)

    i = open(picfile, 'rb')
    img = i.read()

    print("正在识别图片：\t" + filename)

    message = client.webImage(img)          # 通用文字识别
    # message = client.basicAccurate(img)   # 通用文字高精度识别
    i.close()
    if len(message['words_result']):
        word = message.get('words_result')[0].get('words')
        return word
    else:
        return None
