dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from torchcnn14.cnn14 import Cnn14

model_urls = {
    'cnn14_16k': 'https://github.com/HarlandZZC/torchcnn14/'
              'releases/download/v0.1/cnn14_16k.pth',
    'cnn14_32k': 'https://github.com/HarlandZZC/torchcnn14/'
           'releases/download/v0.1/cnn14_32k.pth'
}

def Cnn14(**kwargs):
    model = Cnn14(urls=model_urls, **kwargs)
    return model