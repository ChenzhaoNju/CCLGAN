#Cycle Contrastive Adversarial Learning with Structural Consistency for Unsupervised High-Quality Image Deraining Transformer
<hr>
<i> In this paper, we develop a new
cycle contrastive learning generative adversarial framework for
unsupervised SID, which mainly consists of location contrastive
learning (LCL) and cycle contrastive learning (CCL). Specifically,
LCL implicitly constrains the mutual information of the same location of different exemplars to maintain the content information.
Meanwhile, CCL achieves high-quality semantic reconstruction and
rain-layer stripping by pulling similar features together while pushing dissimilar features further in both semantic and discriminative
spaces. Apart from contrastive learning, we attempt to introduce
vision transformer (VIT) into our network architecture to further
improve the performance. Furthermore, to obtain a stronger representation,we propose a multi-layer channel compression attention
module (MCCAM) to extract a richer attention map. Equipped with
the above techniques, our proposed unsupervised SID algorithm,
called CCLformer, can show advantageous image deraining performance. Extensive experiments demonstrate both the superiority of
our method and the effectiveness of each module in CCLformer.</i>

## Package dependencies
The project is built with PyTorch 1.6.0, Python3.6. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```
## Pretrained model
The pre-trained models of CCLGAN on RainCityscapes are provided in checkpoints/ccl. 
## Training
To train CCLGAN on raincityscapes, you can begin the training by:
```train
python train.py --dataroot DATASET_ROOT  --name NAME --gpu_ids xxx
```

To train CCLformer on raincityscapes, you can begin the training by:
```train
python train.py --dataroot DATASET_ROOT  --name NAME --gpu_ids xxx --netG HTG
```

The DATASET_ROOT example are provided in datasets/raincityscape.
## Evaluation
To evaluate CCLGAN, you can run:
```test
python test.py --dataroot DATASET_ROOT  --name NAME  --gpu_ids xxx
```

To evaluate CCLformer, you can run:
```test
python test.py --dataroot DATASET_ROOT  --name NAME  --gpu_ids xxx --netG HTG
```

## Acknowledgement
This code is inspired by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

