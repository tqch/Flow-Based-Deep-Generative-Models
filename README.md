# Flow-based Deep Generative Models

This is the course final project for SDS 384 Statistical Maching Learning Optimization.

## Models implemented (current):
- Basic flows (planar & radial)
- RealNVP (adapted from the original TensorFlow implementation [[link]](https://git.dst.etit.tu-chemnitz.de/external/tf-models/-/tree/master/research/real_nvp))

## RealNVP Usage:
```
# mnist
python train --dataset mnist --hidden_dim 32 --num_levels 2 --num_residual_blocks 5
# cifar10
python train --dataset cifar10 --hidden_dim 64 --num_levels 2 --num_residual_blocks 8
# celeba
python train --dataset celeba --hidden_dim 32 --num_levels 5 --num_residual_blocks 2
```

## References:

- Rezende, D. &amp; Mohamed, S.. (2015). Variational Inference with Normalizing Flows. <i>Proceedings of the 32nd International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 37:1530-1538 Available from https://proceedings.mlr.press/v37/rezende15.html.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using Real NVP. <i>5th International Conference on Learning Representations</i>, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Opgehaal van https://openreview.net/forum?id=HkpbnH9lx
- van den Berg, R., Hasenclever, L., Tomczak, J., & Welling, M. (2018). Sylvester normalizing flows for variational inference. <i>proceedings of the Conference on Uncertainty in Artificial Intelligence</i> (UAI). http://auai.org/uai2018/proceedings/papers/156.pdf
- Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, & R. Garnett (Reds), <i>Advances in Neural Information Processing Systems</i> (Vol 31). Opgehaal van https://proceedings.neurips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. <i>International Conference on Learning Representations</i>. https://openreview.net/forum?id=rJxgknCcK7
