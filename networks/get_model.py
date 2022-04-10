import networks


def get_net(model_name, in_c=3, num_class=6, img_size=512, pretrained_path=None):
    # try:
    #     model_class = getattr(networks, model_name)
    #     model = model_class(in_c, num_class)
    #     return model
    # except:
    #     print('this model is not exist!!!!, Now supported the base U_Net model ')
    #     model_class = getattr(networks, 'U_Net')
    #     model = model_class(in_c, num_class)
    #     return model
    model_class = getattr(networks, model_name)
    model = model_class(in_c, num_class, pretrained_path=pretrained_path)
    return model
