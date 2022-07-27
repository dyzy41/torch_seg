import networks


def get_net(model_name, in_c=3, num_class=6, img_size=512, pretrained_path=None):
    model_class = getattr(networks, model_name)
    model = model_class(in_c=in_c, num_class=num_class, pretrained_path=pretrained_path)
    return model