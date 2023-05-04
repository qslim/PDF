# Towards Better Graph Representation Learning with <u>P</u>arameterized <u>D</u>ecomposition & <u>F</u>iltering

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper "Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering".

## Requirements

The following packages need to be installed:

- `pytorch==1.13.0`
- `dgl==0.9.1`
- `ogb==1.3.5`
- `numpy`
- `easydict`
- `tensorboard`
- `tqdm`
- `json5`

## Usage

#### ZINC
- Change your current directory to [zinc](zinc);
- Download the dataset: `sh script_download_dataset.sh`;
- Configure hyper-parameters in [ZINC.json](zinc/ZINC.json);
- Start training: `sh run_script.sh`.

#### ogbg-molpcba
- Change your current directory to [ogbg/mol](ogbg/mol);
- Configure hyper-parameters in [ogbg-molpcba.json](ogbg/mol/ogbg-molpcba.json).json);
- Start training: `sh run_script.sh`.

#### TUDataset
- Change your current directory to [tu](tu);
- configure hyper-parameters in [configs/\<dataset\>.json](tu/configs);
- Set dataset name in [run_script.sh](tu/run_script.sh);
- Start training: `sh run_script.sh`.

## Reference
```
@inproceedings{yang2023towards,
  title = {Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering},
  author = {Mingqi Yang and Wenjie Feng and Yanming Shen and Bryan Hooi},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year = {2023},
}
```

## License

[MIT License](LICENSE)
