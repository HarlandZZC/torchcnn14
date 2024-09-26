dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from torchcnn14.cnn14 import cnn14

model_urls = {
    'cnn14_16k': 'https://github.com/HarlandZZC/torchcnn14/'
              'releases/download/v0.1/cnn14_16k.pth',
    'cnn14_32k': 'https://github.com/HarlandZZC/torchcnn14/'
           'releases/download/v0.1/cnn14_32k.pth'
}

def vggish(**kwargs):
    model = cnn14(urls=model_urls, **kwargs)
    return model