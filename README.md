# Flow-based Deep Generative Models

This is the course final project for SDS 384 Statistical Maching Learning Optimization.

## Models implemented (currently):
- Basic flows (planar & radial)
- RealNVP (adapted from the original TensorFlow implementation [[link]](https://git.dst.etit.tu-chemnitz.de/external/tf-models/-/tree/master/research/real_nvp))

## RealNVP Usage:
```
# mnist
python train.py --dataset mnist --hidden_dim 32 --num_levels 2 --num_residual_blocks 5
# cifar10
python train.py --dataset cifar10 --hidden_dim 64 --num_levels 2 --num_residual_blocks 8
# celeba
python train.py --dataset celeba --hidden_dim 32 --num_levels 5 --num_residual_blocks 2
```

## Notes on CelebA dataset
It has been a well-known issue that first-time download of CelebA via `torchvision` API, i.e. `torchvision.datasets.CelebA` will fail almost surely with option `download=True` since Google imposed their restriction on data access and daily limit. One solution is to directly open the dataset's official Google Drive link [[here]](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing) in a browser and download it. Alternatively, we may resort to the `kaggle` datasets API. To use this solution, please set `FROM_KAGGLE` in the `datasets.py` to `True`. Plus, you will need a Kaggle account and an authentication token. 

## References:

- Rezende, D. &amp; Mohamed, S.. (2015). Variational Inference with Normalizing Flows. <i>Proceedings of the 32nd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 37:1530-1538 Available from https://proceedings.mlr.press/v37/rezende15.html.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using Real NVP. <i>5th International Conference on Learning Representations</i>, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Opgehaal van https://openreview.net/forum?id=HkpbnH9lx
- van den Berg, R., Hasenclever, L., Tomczak, J., & Welling, M. (2018). Sylvester normalizing flows for variational inference. <i>proceedings of the Conference on Uncertainty in Artificial Intelligence</i> (UAI). http://auai.org/uai2018/proceedings/papers/156.pdf
- Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, & R. Garnett (Reds), <i>Advances in Neural Information Processing Systems</i> (Vol 31). Opgehaal van https://proceedings.neurips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. <i>International Conference on Learning Representations</i>. https://openreview.net/forum?id=rJxgknCcK7

