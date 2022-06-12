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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=0)
    parser.add_argument("--num_residual_blocks", type=int, default=0)
    parser.add_argument("--img_save_dir", type=str, default="./figs")
    parser.add_argument("--model_save_dir", type=str, default="./models")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--chkpt-intv", type=int, default=5)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # default (hype-)parameters

    print(f"Loading model configs of {args.dataset}...")

    if args.dataset == "mnist":
        channels = 1
        image_size = 32
        hidden_dim = 32
        num_levels = 2
        num_residual_blocks = 5
    elif args.dataset == "cifar10":
        channels = 3
        image_size = 32
        hidden_dim = 64
        num_levels = 2
        num_residual_blocks = 8
    elif args.dataset == "celeba":
        channels = 3
        image_size = 64
        hidden_dim = 32
        num_levels = 5
        num_residual_blocks = 2
    else:
        raise NotImplementedError

    if args.num_levels <= 0:
        print("num_levels")
        print("\tNon-positive input detected!")
        print(f"\tUsing default value for {args.dataset}: {num_levels}")
    else:
        num_levels = args.num_levels
    if args.num_residual_blocks <= 0:
        print("num_residual_blocks")
        print("\tNon-positive input detected!")
        print(f"\tUsing default value for {args.dataset}: {num_residual_blocks}")
    else:
        num_residual_blocks = args.num_residual_blocks
    if args.hidden_dim <= 0:
        print("hidden_dim")
        print("\tNon-positive input detected!")
        print(f"\tUsing default value for {args.dataset}: {hidden_dim}")
    else:
        hidden_dim = args.hidden_dim

    train_loader = get_data(root, dataset=args.dataset, download=args.download, batch_size=args.batch_size)

    input_shape = (channels, image_size, image_size)
    factor_out = 0.5
    l2_reg_coef = 5e-5
    sample_shape = (
        channels * 4 ** (num_levels - 1),
        image_size // 2 ** (num_levels - 1),
        image_size // 2 ** (num_levels - 1)
    ) if factor_out is None else input_shape

    img_save_dir = os.path.join(args.img_save_dir, args.dataset)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    model_save_dir = args.model_save_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model = RealNVP(input_shape, num_residual_blocks=num_residual_blocks,
                    hidden_dim=hidden_dim, num_levels=num_levels, factor_out=factor_out)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    chkpt_path = os.path.join(model_save_dir, f"{args.dataset}_realnvp.pt")
    if args.restart:
        try:
            chkpt = torch.load(chkpt_path, map_location=device)
            model.load_state_dict(chkpt["model"])
            optimizer.load_state_dict(chkpt["optimizer"])
            start_epoch = chkpt["epoch"]
            del chkpt
        except FileNotFoundError:
            print("Checkpoint file does not exists!")
            print("Training from scratch...")

    # fixed latent code
    z = torch.randn((64,) + sample_shape)
    n_epochs = args.n_epochs
    for e in range(start_epoch, n_epochs):
        with tqdm(train_loader, desc=f"{e + 1}/{n_epochs} epochs") as t:
            train_neg_logp = 0
            train_total = 0
            model.train()
            for i, (x, _) in enumerate(t):
                if args.dry_run and i < (len(train_loader) - 1):
                    continue
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
                if i == len(train_loader) - 1:
                    model.eval()
                    with torch.no_grad():
                        x = model.backward(z.to(device))
                    x = pixel_transform(x, inverse=True)
                    img = make_grid(x.cpu(), nrow=8)
                    _ = plt.imsave(
                        os.path.join(img_save_dir, f"realnvp_{args.dataset}_epoch_{e + 1}.png"),
                        img.numpy().transpose(1, 2, 0)
                    )
                    # save a checkpoint every {args.chkpt_intv} epochs
                    if (e + 1) % args.chkpt_intv == 0:
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": e + 1
                        }, chkpt_path)
