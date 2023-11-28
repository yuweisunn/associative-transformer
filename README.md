# Associative Transformer
More details will be added soon.

["Associative Transformer Is A Sparse Representation Learner"](https://arxiv.org/abs/2309.12862)
NeurIPS 2023 Associative Memory & Hopfield Networks

## Training

### Default on CIFAR10
      python ait.py

### Customized Settings
      python ait.py --warmup_t $warmup_t --beta $beta --pattern_size $pattern_size --memory_dim $memory_dim --bottleneck $bottleneck --dataset $dataset --epochs $epochs --batch_size $batch_size --patch_size $patch_size --model_size $model_size
      

## Citation

```
@article{sun2023associative,
  author = {Yuwei Sun, Hideya Ochiai, Zhirong Wu, Stephen Lin, Ryota Kanai},
  title = {Associative Transformer is a Sparse Representation Learner},
  journal = {arXiv preprint:2309.12862},
  year = {2023}
}
```
