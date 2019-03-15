# 3D Object Detection

## Getting Started
Implemented and tested on Ubuntu 16.04 with Python 3.5.5 and Tensorflow 1.3.0.

* Clone this repository 	
```
git clone git@github.com:leofansq/3D-Detection.git
```

* Compile C++ files in [wavedata](/wavedata)
```
cd wavedata/tools/core/lib
cmake src
make
```

* Add to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PWD/wavedata
```

* We use Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level avod folder):
```
sh avod/protos/run_protoc.sh
```
## Training

### Dataset
To train on the [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

* Download the data and place it in your home folder at `~/Kitti/object`
* Go [here](https://drive.google.com/open?id=1yjCwlSOfAZoPNNqoMtWfEjPCfhRfJB-Z) and download the `train.txt`, `val.txt` and `trainval.txt` splits into `~/Kitti/object`. Also download the `planes` folder into `~/Kitti/object/training`

The folder should look something like the following:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```
### Generate planes yourself
You can generate the plane files by yourself using RANSAC. The repository [Tools _make _planes](https://github.com/leofansq/Tools_make_planes) is to generate these files.

### Mini-batch Generation
The training data needs to be pre-processed to generate mini-batches for the RPN. To configure the mini-batches, you can modify `avod/configs/mb_preprocessing/rpn_[class].config`. You also need to select the *class* you want to train on. Inside the `scripts/preprocessing/gen_mini_batches.py` select the classes to process. By default it processes the *Car* and *People* classes, where the flag `process_[class]` is set to True. The People class includes both Pedestrian and Cyclists. You can also generate mini-batches for a single class such as *Car* only.

```
cd avod
python scripts/preprocessing/gen_mini_batches.py
```

Once this script is done, you should now have the following folders inside `avod/data`:
```
data
    label_clusters
    mini_batches
```
### Training Configuration
There are sample configuration files for training inside `avod/configs`. You can train on the example configs, or modify an existing configuration. To train a new configuration, copy a config, e.g. `example.config`, rename this file to a unique experiment name and make sure the file name matches the `checkpoint_name: 'example'` entry inside your config.

### Run Trainer
To start training, run the following:

```
python avod/experiments/run_training.py --pipeline_config=avod/configs/example.config  --device='0' --data_split='train'
```

> If the process was interrupted, training (or evaluation) will continue from the last saved checkpoint if it exists.

## Evaluation
### Run Evaluator
To start evaluation, run the following:

```
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/example.config --device='0' --data_split='val' --ckpt_start='60000'
```

> The *ckpt_start* is the number of the model which you want to start evaluation. That means you can start your evaluation from any model, instead of starting from the begining.

### View TensorBoard Summaries
To view the TensorBoard summaries:

```
cd avod/data/outputs/example
tensorboard --logdir logs
``` 

### Results
The evaluator runs the KITTI native evaluation code on each checkpoint. Predictions are converted to KITTI format and the AP is calculated for every checkpoint. 

To see the result txt:

```
cd avod/data/outputs/example/predictions/kitti_native_eval
sh all_eval.sh 0.1
```
> *0.1* is the score threshold. IoUs are set to (0.7, 0.5, 0.5) .

## Inference
### Run Inference
To run inference, run the following script:

```
python avod/experiments/run_inference.py --checkpoint_name='example' --data_split='val' --ckpt_indices=120 --device='0'
```

> The *ckpt_indices* here indicates the indices of the checkpoint in the list. You can also just set this to *-1* to evaluate the last checkpoint.

### KITTI's Format
If you want to submit your result to KITTI sever, you need to transform the format to KITTI's format. Modify *checkpoint_name* and *global_steps* in [save_kitti _predictions.py](/scripts/offline_eval/save_kitti_predictions.py). Run the following script:

```
python save_kitti_predictions.py
```

## Visualization
### 3D Detection Results
To visualize the 3D detection results, you can run [show_predictions _2d.py](/demos/show_predictions_2d.py). You need to configure the parameters to your specific experiments, and run the following script:

```
python show_predictions_2d.py
```

### BEV Detection Results
To visualize the BEV detection results, you can run [show_predictions _bev.py](/demos/show_predictions_bev.py). You need to configure the parameters to your specific experiments, and run the following script:

```
python show_predictions_bev.py
```

### AP Information
The [plot_ap.py](/scripts/offline_eval/plot_ap.py) will plot the AP vs. step, and print the 5 highest performing checkpoints for each evaluation metric at the moderate difficulty. You need to configure the parameters to your specific experiments, and run the following script:

```
python plot_ap.py
```






