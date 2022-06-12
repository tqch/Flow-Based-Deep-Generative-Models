import os
import json
import torch
import numpy as np


def pixel_transform(x, alpha=0.05, inverse=False):
    if not inverse:
        x = torch.logit(alpha*0.5+(1-alpha)*x)
    else:
        x = (torch.sigmoid(x)-alpha*0.5)/(1-alpha)
        x = x.clip(0, 1)
    return x


def bisect(func, fx, x_min, x_max, max_iter=100, tol=1e-6):
    # bisection solver
    left = x_min
    right = x_max
    mid = np.zeros(len(fx))

    f_left = func(left) - fx
    f_right = func(right) - fx

    update_indices = np.arange(len(fx))

    for it in range(max_iter):
        if it == 0:
            assert np.all(np.sign(f_left) != np.sign(f_right)), \
                "Function values of x_min and x_max must have different sign!"
        mid[update_indices] = (left[update_indices] + right[update_indices]) / 2
        f_mid = func(mid[update_indices]) - fx[update_indices]
        # whether the left points have the same signs of function values as the midpoints
        same_sign = np.sign(f_left) == np.sign(f_mid)
        # the left points that need to be replaced by the midpoints
        move2right = update_indices[same_sign]
        move2left = update_indices[~same_sign]
        # update
        left[move2right] = mid[move2right]
        f_left[move2right] = f_mid[same_sign]
        right[move2left] = mid[move2left]
        f_right[move2left] = f_mid[~same_sign]
        update_indices = update_indices[np.abs(f_mid) > tol]
        if len(update_indices) == 0:
            # jump out of the loop if no midpoint needs to be updated
            break
    return mid


def kaggle_setup():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    user_auth_token = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(user_auth_token):
        choice = input("User authentication token does not exists! Continue? [y/n]").strip().lower()
        while choice not in ["y", "n"]:
            choice = input("User authentication token does not exists! Continue? [y/n]").strip().lower()
        if choice == "y":
            try:
                os.mkdir(kaggle_dir)
            except FileExistsError:
                pass
            print(f"Kaggle directory is located at {kaggle_dir}.")
            username = input("Username: ")
            key = input("Key: ")
            def opener(path, flags): return os.open(path, flags, mode=0o600)
            with open(user_auth_token, "w", opener=opener) as f:
                json.dump({"username": username, "key": key}, f)
