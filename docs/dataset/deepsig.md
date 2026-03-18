It is recommended to symlink the dataset root to `$CSRR/data`. If your folder structure is different, you may need to
change the corresponding paths in config files.

```
ChangShuoRadioRecognition
├── configs
├── data
│   ├── ModulationClassification
│   │   ├── DeepSig
│   │   │   ├── 201610A
│   │   │   │   ├── train.json
│   │   │   │   ├── val.json
│   │   │   │   ├── test.json
│   │   │   │   ├── sequence_data
│   │   │   │   │   ├── iq
│   │   │   │   │   ├── ap
│   │   │   │   ├── constellation_data
│   │   │   │   │   ├── filter_size_0.010_stride_0.005
│   ├── SignalSeparation
│   │   ├── CSRR
│   │   │   ├── qpsk_16qam
│   │   │   │   ├── complex
│   │   │   │   │   ├── train_data.mat
│   │   │   │   │   ├── val_data.mat
│   │   │   │   │   ├── test_data.mat
│   │   │   │   ├── real
│   │   │   │   │   ├── train_data.mat
│   │   │   │   │   ├── val_data.mat
│   │   │   │   │   ├── test_data.mat

```

The deepsig data have to be converted into the specified format using `tools/convert_datasets/comvert_deepsig.py`: