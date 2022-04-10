import timm
import torch

def get_model(model_name, in_c=3, pretrained=False, checkpoint_path='None'):
    if pretrained:
        print('loading pretrained model {}, from the default path'.format(checkpoint_path))
        model = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=in_c)
        return model
    elif 'pth' in checkpoint_path:
        print('loading pretrained model {}'.format(checkpoint_path))
        model = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=in_c)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # del state_dict['fc.weight']
        # del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict=False)
        return model
    else:
        print('no pretrained model')
        model = timm.create_model(model_name, pretrained=pretrained, features_only=True, in_chans=in_c)
        return model