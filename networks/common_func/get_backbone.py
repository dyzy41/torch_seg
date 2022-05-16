import timm
import torch

def get_model(model_name, in_c=3, checkpoint_path='None'):
    if checkpoint_path == 'download':
        print('download the pretrained model')
        model = timm.create_model(model_name, pretrained=True, features_only=True, in_chans=in_c)
        return model
    elif checkpoint_path == '':
        print('do not use the pretrained model')
        model = timm.create_model(model_name, pretrained=False, features_only=True, in_chans=in_c)
        return model
    else:
        print('loading pretrained model {}'.format(checkpoint_path))
        model = timm.create_model(model_name, pretrained=False, features_only=True, in_chans=in_c)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return model
