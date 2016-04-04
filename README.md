Training network:

```
python src/arrows_train.py <data_dir> <save_dir> <logs_dir>
```

Maximizing layer output:

```
python src/arrows_show_layer.py <model_path> <layer_name> <channel>
```

Example: `python src/arrows_show_layer.py save_arrows/model.ckpt conv2/Conv/BiasAdd 17`
