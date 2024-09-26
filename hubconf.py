dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from torchcnn14.cnn14 import cnn14

model_urls = {
    'cnn14_16k': 'https://github.com/HarlandZZC/cnn14_torch/'
              'releases/download/v0.1/Cnn14_16k_mAP=0.438.pth',
    'cnn14_32k': 'https://github.com/harritaylor/cnn14_torch/'
           'releases/download/v0.1/Cnn14_mAP=0.431.pth'
}

def vggish(**kwargs):
    model = cnn14(urls=model_urls, **kwargs)
    return model