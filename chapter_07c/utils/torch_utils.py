import torch
def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device('cpu')
    else:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')

        if torch.cuda.device_count() > 1:
            device = torch.device('cuda' if cuda else 'cpu')
            print('Found %g GPUs' % torch.cuda.device_count())

    print('Using %s %s\n' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))
    return device
