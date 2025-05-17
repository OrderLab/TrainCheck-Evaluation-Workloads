Original Issue Post: https://github.com/open-mmlab/mmpretrain/issues/702

# MMPretrain-702

**Dependencies:**
- pytorch 
- mmclassification v0.20.1
- pytorch 1.8.0

Installation Steps:
1. Create a virtual env with Python 3.8.12.
```bash
conda create -n mm702 python=3.8.12
```
2. Install dependencies.
``` bash
# install pytorch 1.8.0
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# install mmclassification v0.20.1
mkdir deps; cd deps
git clone git@github.com:open-mmlab/mmpretrain.git; cd mmpretrain
git checkout a7f8e96
pip3 install -e .

# install mmcv manually
pip3 install -r requirements/mminstall.txt 

# a too up-to-date version of yapf might be installed
pip3 install "yapf==v0.32.0"
```

3. Run mmpretrain with `frozen_stages` being a non-zero value.
```bash
bash tools/dist_train.sh configs/tutorial/vgg19-finetune-cifar.py 4
```

## Root Cause
`self._freeze_stages` must be called before doing forward, if `frozen_stages > 0`. 
There are two typical places in mmpretrain where this function gets called:
1. `self.__init__`: freeze the stages at model initialization time
2. `self.train`: freeze the stages when the model is explicitly set to training model.

For most models, MMPretrain only does 2 (except for ResNet, RegNet, and ConvNeXt).
Thus, `model.train` must be called explicitly prior to training to make sure `frozen_stages` is configured correctly.

## Detection with Invariants
There can be multiple invariants for the diagnosis.
1. APISeq: ensure that `self._freeze_stages` or `self.train` is invoked prior to `self.forward`.
2. EventContain: ensure that `self.__init__` contains invocation of `self._freeze_stages()`.

**Input**: inv 1's precondition is a little bit hard to infer (i.e. only self.forward in training mode has to be preceded by a `self._freeze_stages`). 
inv2 is easier to infer. Since only ResNet, RegNet, and ConvNeXt calls `self._freeze_stages` inside init, the only correct pipelines to infer invariant from would be using either
ResNet, RegNet or ConvNeXt (as all other models will suffer from the same exception).






