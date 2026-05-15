import torch


def save_checkpoint(path, epoch, generator, discriminator, opt_g, opt_d):
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, path)


def load_checkpoint(path, generator, discriminator, opt_g, opt_d, device):
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt["epoch"] + 1
