# Joint Slot Filling and Intent Detection via Capsule Neural Networks

This project provides tools for joint slot filling and intent detection via Capsule Neural Networks. 

Details about Capsule-NLU can be accessed [here](http://arxiv.org/abs/1812.09471), and the implementation is based on the Tensorflow library. 

## Quick Links
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

For training, a GPU is recommended to accelerate the training speed. 

### Tensorflow

The code is based on Tensorflow 1.5. You can find installation instructions [here](https://www.tensorflow.org/install).

### Dependencies

The code is written in Python 3.5. Its dependencies are summarized in the file ```requirements.txt```. 

tensorflow_gpu==1.5.0

numpy==1.14.0

six==1.11.0

scikit_learn==0.21.2

You can install these dependencies like this:
```
pip3 install -r requirements.txt
```
## Usage
* Run the full model on SNIPS-NLU dataset with default hyperparameter settings<br>
```python3 train.py --dataset=snips```<br>
    > Try run without early-stop
    ```python3 train.py --dataset=snips --no_early_stop --max_epochs=60```

* Run the model without re-routing on SNIPS-NLU dataset<br>
```python3 train.py --dataset=snips --model_type=without_rerouting```

* For all available hyperparameter settings, use<br>
```python3 train.py -h```

## Data
### Format
Each dataset is a folder under the ```./data``` folder, where each sub-folder indicates a train/valid/test split:
```
./data
└── snips
    ├── test
    │   ├── label
    │   ├── seq.in
    │   └── seq.out
    ├── train
    │   ├── label
    │   ├── seq.in
    │   └── seq.out
    └── valid
        ├── label
        ├── seq.in
        └── seq.out
```
In each sub-folder,<br> 
* ```label``` file contains the intent label.<br> 
    e.g. ```AddToPlaylist```

* ```seq.in``` file contains utterances as the input sequences. Each line indicates one utterance and words are separated by a single space.<br>
    e.g. ```add sabrina salerno to the grime instrumentals playlist```

* ```seq.out``` file contains ground truth slot labels. Each line indicates a sequence of slot labels and the [BIO tagging scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) is used.<br>
    e.g. ```O B-artist I-artist O O B-playlist I-playlist O```

### Work on your own data
Prepare and organize your dataset in a folder according to the [format](#format) and put it under ```./data/``` and use `--dataset=foldername` during training. 

For example, your dataset is `./data/mydata`, then you need to use the flag `--dataset=mydata` for ```train.py```.<br>
Your dataset should be seperated to three folders - train, test, and valid, which is named 'train', 'test', and 'valid' by default setting of train.py. 
Each of these folders contain three files - word sequence, slot label, and intent label, which is named 'seq.in', 'seq.out', and 'label' by default setting of train.py. 
  
## Results
| Model                                     |                  |    SNIPS-NLU             |                |          |      ATIS            |                |
|-------------------------------------------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|                                           |    Slot (F1)   |  Intent (Acc)  |  Overall (Acc) |    Slot (F1)   |  Intent (Acc)  |  Overall (Acc) |
| CNN TriCRF                                |        -       |        -       |        -       |      0.944     |        -       |        -       |
| [Joint Seq](https://github.com/yvchen/JointSLU)        |      0.873     |      0.969     |      0.732     |      0.942     |      0.926     |      0.807     |
| [Attention BiRNN](https://github.com/HadoopIt/rnn-nlu)  |      0.878     |      0.967     |      0.741     |      0.942     |      0.911     |      0.789     |
| [Slot-Gated Full Atten.](https://github.com/MiuLab/SlotGated-SLU) |      0.888     |      0.970     |      0.755     |      0.948     |      0.936     |      0.822     |
| [DR-AGG](https://github.com/FudanNLP/Capsule4TextClassification)         |        -       |      0.966     |        -       |        -       |      0.914     |        -       |
| [IntentCapsNet](https://github.com/congyingxia/ZeroShotCapsule)          |        -       |      **0.974**     |        -       |        -       |      0.948     |        -       |
| Capsule-NLU (our)                         | **0.918** |      0.973     | **0.809** | **0.952** | **0.950** | **0.834** |
 
 
## Acknowledgements
https://github.com/MiuLab/SlotGated-SLU

https://github.com/FudanNLP/Capsule4TextClassification

https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines

## Reference
```
@inproceedings{zhang2019joint,
  title={Joint slot filling and intent detection via capsule neural networks},
  author={Zhang, Chenwei and Li, Yaliang and Du, Nan and Fan, Wei and Yu, Philip S},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}
}
```
