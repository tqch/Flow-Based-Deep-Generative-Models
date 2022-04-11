if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import torch

    from torch.optim import Adam
    from flows.realnvp import RealNVP
    from torchvision.utils import make_grid
    from torch.nn.utils import clip_grad_norm_

    from utils import pixel_transform
    from datasets import get_data

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["figure.dpi"] = 144

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="~/datasets")
    parser.add_device("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"])
    parser.add_argument("--batch_size", type=str, default=64)
    parser.add_argument("--num_residual_blocks", type=int, default=5)
    parser.add_argument("--img_save_dir", type=str, default="./figs")
    parser.add_argument("--model_save_dir", type=str, default="./models")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=30)
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.datasets == "mnist":
        channels = 1
        image_size = 32
        hidden_dim = 32
        num_levels = 2
        num_residual_blocks = 5
    elif args.datasets == "cifar10":
        channels = 3
        image_size = 32
        hidden_dim = 64
        num_levels = 2
        num_residual_blocks = 8
    elif args.datasets == "celeba":
        channels = 3
        image_size = 64
        hidden_dim = 32
        num_levels = 5
        num_residual_blocks = 2
    else:
        raise NotImplementedError
    train_loader = get_data(root, dataset=args.dataset, batch_size=args.batch_size, num_workers=4)

    input_shape = (channels, image_size, image_size)
    factor_out = 0.5
    l2_reg_coef = 5e-5
    sample_shape = (
        channels * 4 ** (num_levels - 1),
        image_size // 2 ** (num_levels - 1),
        image_size // 2 ** (num_levels - 1)
    ) if factor_out is None else input_shape

    img_save_dir = args.img_save_dir
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    model_save_dir = args.model_save_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model = RealNVP(input_shape, num_residual_blocks=num_residual_blocks,
                    hidden_dim=hidden_dim, num_levels=num_levels, factor_out=factor_out)
    model.to(device)

    n_epochs = args.n_epochs

    optimizer = Adam(model.parameters(), lr=0.001)

    for e in range(n_epochs):
        with tqdm(train_loader, desc=f"{e + 1}/{n_epochs} epochs") as t:
            train_neg_logp = 0
            train_total = 0
            model.train()
            for i, (x, _) in enumerate(t):
                x = pixel_transform(x)
                _, neg_logp = model(x.to(device))
                l2_reg = model.l2_reg()
                loss = neg_logp + l2_reg_coef * l2_reg
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=100)
                optimizer.step()
                train_neg_logp += neg_logp.item() * x.size(0)
                train_total += x.size(0)
                t.set_postfix({"train_neg_logp": train_neg_logp / train_total})
                if i == len(train_loader)-1:
                    model.eval()
                    with torch.no_grad():
                        z = torch.randn((16,) + sample_shape)
                        x = model.backward(z.to(device))
                    x = pixel_transform(x, inverse=True)
                    plt.figure(figsize=(16, 16))
                    grid = make_grid(x.cpu(), nrow=4)
                    _ = plt.imshow(grid.numpy().transpose(1, 2, 0))
                    plt.savefig(os.path.join(img_save_dir, f"realnvp_celeba_epoch_{e+1}.png"))
                    plt.close()
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_save_dir, "models/celeba_realnvp.pt")
                    )

