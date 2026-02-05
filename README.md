# Spherical CNN JAX Library

This repo contains a JAX library to implement spherical CNNs and spin-weighted
spherical CNNs. It was used in our ICML 2023 paper "[Scaling Spherical
CNNs](https://arxiv.org/abs/2306.05420)." The code can also reproduce results
from our previous papers "[Spin-Weighted Spherical
CNNs](http://arxiv.org/abs/2006.10731)", NeurIPS'20 and "[Learning SO(3)
Equivariant Representations with Spherical
CNNs](https://arxiv.org/pdf/1711.06721)", ECCV'18.

## Experiments

### QM9

Use the following instructions to launch a short training job on QM9/H. See
[default.py](https://github.com/google-research/spherical-cnn/blob/main/spherical_cnn/molecules/configs/default.py)
for the longer configurations that reproduce the results in the paper.

```shell
git clone https://github.com/google-research/spherical-cnn.git
cd spherical-cnn
# Create a docker container, download and install dependencies.
docker build -f dockerfile-qm9 -t spherical_cnn_qm9 .
# Start training.
docker run spherical_cnn_qm9 \
    --workdir=/tmp/training_logs \
    --config=spherical_cnn/spherical_mnist/configs/small.py \
    --config.per_device_batch_size=2
```

It should train at around 21.9 steps/s with batch size 2 on 8 V100s and reach
around 10.83 meV error for the enthalpy of atomization H (this trains for 250
epochs while 5.69 meV error in the paper was obtained by training for 2000
epochs, see
[default.py](https://github.com/google-research/spherical-cnn/blob/main/spherical_cnn/molecules/configs/default.py)).

### Spherical MNIST

Use the following instructions to train a small model on GPU on the spherical
MNIST dataset.

```shell
git clone https://github.com/google-research/spherical-cnn.git
cd spherical-cnn
# Create a docker container, download and install dependencies, download and
# process the dataset.
docker build -f dockerfile-spherical-mnist -t spherical_cnn_mnist .
# Start training.
docker run spherical_cnn_mnist \
    --workdir=/tmp/training_logs \
    --config=spherical_cnn/spherical_mnist/configs/default.py \
    --config.model_name=spin_classifier_6_layers \
    --config.dataset=spherical_mnist/rotated \
    --config.combine_train_val_and_eval_on_test
```

It should train at around 22 steps/s on a single P100 and reach around 99.5%
accuracy. Outputs should look something like this:

```
INFO 2023-08-21T19:30:28.855726181Z [Hyperparameters] {'checkpoint_every_steps': 1000, 'combine_train_val_and_eval_on_test': True, 'dataset': 'spherical_mnist/rotated', 'eval_every_steps': 1000, 'eval_pad_last_batch': True, 'learning_rate': 0.001, 'learning_rate_schedule': 'cosine', 'log_loss_every_steps': 100, 'model_name': 'spin_classifier_6_layers', 'num_epochs': 12, 'num_eval_steps': -1, 'num_train_steps': -1, 'per_device_batch_size': 32, 'seed': 42, 'shuffle_buffer_size': 1000, 'trial': 0, 'warmup_epochs': 1, 'weight_decay': 0.0}
INFO 2023-08-21T19:30:28.856940603Z Starting training loop at step 1.
INFO 2023-08-21T19:30:28.857277764Z [1] param_count=39146
INFO 2023-08-21T19:31:12.653463819Z [100] learning_rate=5.333333683665842e-05, loss=2.2819416522979736, loss_std=0.10880677402019501, train_accuracy=0.19312499463558197
INFO 2023-08-21T19:31:29.503783929Z [200] learning_rate=0.00010666667367331684, loss=1.8683496713638306, loss_std=0.14256055653095245, train_accuracy=0.3765625059604645

(...)

INFO 2023-08-21T19:48:41.827125703Z [22400] learning_rate=5.799532232231286e-08, loss=0.006118293385952711, loss_std=0.015895500779151917, train_accuracy=0.9984374642372131
INFO 2023-08-21T19:48:44.634986829Z [22500] steps_per_sec=22.0576
INFO 2023-08-21T19:48:44.635090221Z [22500] uptime=1095.78
INFO 2023-08-21T19:48:44.695150873Z [22500] learning_rate=0.0, loss=0.003276276867836714, loss_std=0.005533786956220865, train_accuracy=0.9993749856948853
INFO 2023-08-21T19:49:00.926620697Z Starting evaluation.
INFO 2023-08-21T19:49:16.283256304Z [22500] accuracy=0.9949000477790833, eval_loss=0.033049359917640686
INFO 2023-08-21T19:49:16.288987270Z Finishing training at step 22500
```

## Unit tests

The code is extensively tested. The snippet below runs all tests given a docker
container created from instructions above.

```shell
docker run --entrypoint pytest -it spherical_cnn -vv spherical_cnn_mnist
```

## References

If you use this code, please cite the papers:

```bibtex
@InProceedings{pmlr-v202-esteves23a,
  title = {Scaling Spherical {CNN}s},
  author = {Esteves, Carlos and Slotine, Jean-Jacques and Makadia, Ameesh},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {9396--9411},
  year = {2023},
  editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  month = {23--29 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v202/esteves23a/esteves23a.pdf},
  url = {https://proceedings.mlr.press/v202/esteves23a.html},
}
```

```bibtex
@inproceedings{esteves20_swscnn,
 author = {Esteves, Carlos and Makadia, Ameesh and Daniilidis, Kostas},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
 pages = {8614--8625},
 publisher = {Curran Associates, Inc.},
 title = {Spin-Weighted Spherical CNNs},
 url = {https://proceedings.neurips.cc/paper_files/paper/2020/file/6217b2f7e4634fa665d31d3b4df81b56-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

[![Unittests](https://github.com/google-research/spherical-cnn/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/spherical-cnn/actions/workflows/pytest_and_autopublish.yml)

*This is not an officially supported Google product.*
