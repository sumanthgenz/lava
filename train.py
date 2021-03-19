import torch
import torchaudio
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins.training_type.rpc_sequential import RPCSequentialPlugin

import warnings
import glob
from tqdm import tqdm
import wandb
from absl import app, flags

from aai.experimental.sgurram.lava.src.lightning import LAVALightning, EvalLightning
from aai.experimental.sgurram.lava.src.encoder import SupervisedVideoClassifier
from aai.experimental.sgurram.lava.src.references import save_dir


torchaudio.set_audio_backend("sox_io")
warnings.filterwarnings("ignore")

#Define modes
flags.DEFINE_string('mode',
                  default='train',
                  help='train or test mode',)

flags.DEFINE_string('log',
                  default='gradients',
                  help='log modes',)

flags.DEFINE_string('backend',
                  default='ddp',
                  help='distributed backend for training',)


#Define pl_trainer params
flags.DEFINE_string('root_dir',
                  default='/home/sgurram/Desktop/video_cave',
                  help='root directory to save checkpoints',)


flags.DEFINE_integer('num_gpus',
                    default=2,
                    help='number of gpus to be used with distributed backend',)

flags.DEFINE_integer('max_epochs',
                    default=100,
                    help='number of epochs for training',)

flags.DEFINE_integer('accum_grad_batches',
                    default=1,
                    help='number of batches between gradient descent steps',)

flags.DEFINE_integer('log_freq',
                    default=10,
                    help='number of batches in between logging gradients',)

def train_lava(logging=False):

        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

        hparams = {'gpus':[1], 
                'max_epochs': 100, 
                'auto_lr_find': False,
                'learning_rate': 2e-5,
                'max_lr': 2e-5,
                'min_lr': 4e-6,
                'auto_scale_batch_size': None,
                'batch_size': 8,
                'accumulate_grad_batches': 32,
                'model_dimension': 1024,
                'feature_dimension': 512,
                'seq_len': 256,
                'loss_functions': ['modal_pairwise_nce', 'triplet_centroid_nce'],
                'metrics': ['unimodal_cos_dist', 'modal_pairwise_cos_sim'],
                'num_transformer_heads': 8,
                'num_transformer_layers': 8,
                'dropout': 0.1,
                'overfit_batches': 0,
                'amp_backend': 'native',
                'amp_level': 'O2',
                'precision': 32,
                'log_gpu_memory': 'all',
                'optimizer': 'adamW',
                'scheduler': 'cosine',
                'cosine_steps_period': 4000,
                'warmup': None,
                'profiler': 'simple',
                'distributed_backend': 'ddp',
                'callbacks': [lr_monitor_callback] if logging else None,
                'default_root_dir': '/home/sgurram/Desktop/video_lava',}

        model = LAVALightning(
                model_dimension=hparams['model_dimension'], 
                feat_dimension=hparams['feature_dimension'],
                seqlen=hparams['seq_len'],
                batch_size=hparams['batch_size'], 
                num_heads=hparams['num_transformer_heads'], 
                num_layers=hparams['num_transformer_layers'],
                learning_rate=hparams['learning_rate'],
                min_lr=hparams['min_lr'],
                max_lr=hparams['max_lr'],
                dropout=hparams['dropout'],
                optimizer=hparams['optimizer'],
                scheduler=hparams['scheduler'],
                warmup_mode=hparams['warmup'],
                cosine_steps_period=hparams['cosine_steps_period'])


        if logging:
                wandb_logger = WandbLogger(name='run',project='lava')
                wandb_logger.log_hyperparams(hparams)
                wandb_logger.watch(model, 
                        log='gradients', 
                        log_freq=10)
        else:
                wandb_logger = None

        trainer = pl.Trainer(
                default_root_dir=hparams['default_root_dir'], 
                gpus=hparams['gpus'], 
                max_epochs=hparams['max_epochs'],
                auto_scale_batch_size=hparams['auto_scale_batch_size'],
                auto_lr_find=hparams['auto_lr_find'],
                accumulate_grad_batches=hparams['accumulate_grad_batches'],
                overfit_batches=hparams['overfit_batches'],
                logger=wandb_logger,
                profiler=hparams['profiler'],
                amp_backend=hparams['amp_backend'],
                amp_level=hparams['amp_level'],
                log_gpu_memory=hparams['log_gpu_memory'],
                callbacks=hparams['callbacks'],
                # precision=hparams['precision'],
                # distributed_backend=hparams['distributed_backend'],
                # limit_train_batches=0.01,
                # limit_val_batches=0.01,
                ) 
        
        # trainer.tune(model)

        trainer.fit(model)

def train_classifier(logging=False):

        hparams = {'gpus':[1], 
                'max_epochs': 25, 
                'num_classes': 700,
                'feature_dimension': 512,
                'model_dimension': 1024,
                'num_modalities': 3,
                'batch_size': 12,
                'learning_rate': 1e-3,
                'model_path': '/home/sgurram/Desktop/video_lava/checkpoints/epoch=0-v0.ckpt',
                'accumulate_grad_batches': 16,
                'overfit_batches': 0,
                'type_modalities': 'avt', 
                'modality_blending': 'concat',
                'loss_funtions': ['cross_entropy'],
                'metrics': None,
                'optimizer': 'adam',
                'scheduler': 'n/a',
                'profiler': 'simple',
                'default_root_dir': '/home/sgurram/Desktop/video_lava_classifer',}


        model = EvalLightning(
                num_classes=hparams['num_classes'],
                feature_dimension=hparams['feature_dimension'],
                model_dimension=hparams['model_dimension'],
                num_modalities=hparams['num_modalities'],
                batch_size=hparams['batch_size'],
                learning_rate=hparams['learning_rate'],
                model_path=hparams['model_path'])


        if logging:
                wandb_logger = WandbLogger(name='run',project='lava')
                wandb_logger.log_hyperparams(hparams)
                wandb_logger.watch(model, 
                        log='gradients', 
                        log_freq=10)
        else:
                wandb_logger = None

        trainer = pl.Trainer(
                default_root_dir=hparams['default_root_dir'], 
                gpus=hparams['gpus'], 
                max_epochs=hparams['max_epochs'],
                accumulate_grad_batches=hparams['accumulate_grad_batches'],
                overfit_batches=hparams['overfit_batches'],
                logger=wandb_logger,
                profiler=hparams['profiler']) 

        trainer.fit(model)

def train_supervised_video_encoder():
        # wandb_logger = None

        model = EvalLightning(logger=wandb_logger, classifier=SupervisedVideoClassifier)

        hyperparams = {'gpus':[1], 
                'max_epochs': 25, 
                'batch_size': model.classifier.batch_size,
                'accumulate_grad_batches': 8,
                'learning_rate': model.classifier.learning_rate,
                'feature_dimension': model.classifier.feature_dimension,
                'model_dimension': model.classifier.model_dimension,
                'num_modalities': 3,
                'type_modalities': 'v', 
                 'optimizer': 'Adam',
                'scheduler': 'n/a',}
        

        wandb_logger.watch(model, 
                log='gradients', 
                log_freq=10)

        wandb_logger.log_hyperparams(hyperparams)


        trainer = pl.Trainer(
                default_root_dir='/home/sgurram/Desktop/video_lava_classifier', 
                gpus=[0], 
                max_epochs=25, 
                accumulate_grad_batches=8,
                logger=wandb_logger,
                profiler=True) 
        
        trainer.fit(model)



if __name__ == '__main__':

        FLAGS = flags.FLAGS 
        modes = ["pretrain", "probe", "supervised"]
        mode = 0

        if mode == 0:
                train_lava()
        elif mode == 1:
                train_classifier()
        elif mode == 2:
                train_supervised_video_encoder()
        else:
                print("no modes selected")
