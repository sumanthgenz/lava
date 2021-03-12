import torch
import torchaudio
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins.training_type.rpc_sequential import RPCSequentialPlugin


import warnings
import glob
from tqdm import tqdm
import wandb

from absl import app, flags

torchaudio.set_audio_backend("sox_io")
warnings.filterwarnings("ignore")

from aai.experimental.sgurram.lava.src.lightning import LAVALightning, EvalLightning
from aai.experimental.sgurram.lava.src.encoder import SupervisedVideoClassifier

wandb_logger = WandbLogger(name='run',project='lava')

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

def train_lava():



    # FLAGS for instaniating trainer
    # wandb_logger.watch(model, 
    #         log=FLAGS.log, 
    #         log_freq=FLAGS.log_freq)

    # trainer = pl.Trainer(
    #         default_root_dir=FLAGS.root_dir, 
    #         gpus=FLAGS.num_gpus, 
    #         max_epochs=FLAGS.max_epochs, 
    #         accumulate_grad_batches=FLAGS.accum_grad_batches, 
    #         distributed_backend=FLAGS.backend,
    #         logger=wandb_logger,)   

    #     wandb_logger.watch(model, 
    #         log=FLAGS.log, 
    #         log_freq=FLAGS.log_freq)

 
        lr_monitor_callback = pl.callbacks.LearningRateMonitor()

        hparams = {'gpus':[1], 
                'max_epochs': 100, 
                'auto_lr_find': False,
                'learning_rate': 3e-4,
                'auto_scale_batch_size': None,
                'batch_size': 12,
                'accumulate_grad_batches': 8,
                'model_dimension': 1024,
                'feature_dimension': 512,
                'seq_len': 256,
                'loss_functions': ['modal_pairwise_nce', 'triplet_centroid_nce'],
                'num_transformer_heads': 8,
                'num_transformer_layers': 8,
                'dropout': 0.1,
                'overfit_batches': 0,
                'amp_backend': 'native',
                'amp_level': 'O2',
                'precision': 32,
                'optimizer': 'adamW',
                'scheduler': 'cosine',
                'warmup': None,
                'profiler': True,
                'distributed_backend': 'ddp'}

        model = LAVALightning(dropout=hparams['dropout'],
                        model_dimension=hparams['model_dimension'], 
                        feat_dimension=hparams['feature_dimension'],
                        seqlen=hparams['seq_len'],
                        batch_size=hparams['batch_size'], 
                        num_heads=hparams['num_transformer_heads'], 
                        num_layers=hparams['num_transformer_layers'],
                        learning_rate=hparams['learning_rate'],
                        optimizer=hparams['optimizer'],
                        scheduler=hparams['scheduler'],
                        warmup_mode=hparams['warmup'])

        # wandb_logger = None

        wandb_logger.log_hyperparams(hparams)

        wandb_logger.watch(model, 
        log='gradients', 
        log_freq=10)

        trainer = pl.Trainer(
                default_root_dir='/home/sgurram/Desktop/video_lava', 
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
                # precision=hparams['precision'],
                # distributed_backend=hparams['distributed_backend'],
                # limit_train_batches=0.01,
                # limit_val_batches=0.01,
                # callbacks=[lr_monitor_callback],
                ) 

        trainer.tune(model)
        
        trainer.fit(model)

def train_classifier():
        # wandb_logger = None

        model = EvalLightning(logger=wandb_logger)

        hparams = {'gpus':[1], 
                'max_epochs': 25, 
                'batch_size': model.classifier.batch_size,
                'accumulate_grad_batches': 8,
                'overfit_batches': 0,
                'learning_rate': model.classifier.learning_rate,
                'feature_dimension': model.classifier.feature_dimension,
                'model_dimension': model.classifier.model_dimension,
                'num_modalities': 3,
                'type_modalities': 'avt', 
                 'optimizer': 'Adam',
                'scheduler': 'n/a',}

        trainer = pl.Trainer(
                default_root_dir='/home/sgurram/Desktop/video_lava_classifer', 
                gpus=hparams['gpus'], 
                max_epochs=hparams['max_epochs'],
                accumulate_grad_batches=hparams['accumulate_grad_batches'],
                overfit_batches=hparams['overfit_batches'],
                logger=wandb_logger,
                profiler=True) 
        
        wandb_logger.watch(model, 
                log='gradients', 
                log_freq=10)

        wandb_logger.log_hyperparams(hparams)


        trainer = pl.Trainer(
                default_root_dir='/home/sgurram/Desktop/video_lava_classifier', 
                gpus=hparams['gpus'], 
                max_epochs=hparams['max_epochs'],
                accumulate_grad_batches=hparams['accumulate_grad_batches'],
                overfit_batches=hparams['overfit_batches'],
                logger=wandb_logger,
                profiler=True) 
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
#     train_classifier()
    train_lava()
#     train_supervised_video_encoder()