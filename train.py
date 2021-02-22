import torch
import torchaudio
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import warnings
import glob
from tqdm import tqdm
import wandb

from absl import app, flags

torchaudio.set_audio_backend("sox_io")
warnings.filterwarnings("ignore")

from lightning import LAVALightning, EvalLightning

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
        model = LAVALightning()


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

        # wandb_logger.watch(model, 
        #     log=FLAGS.log, 
        #     log_freq=FLAGS.log_freq)

        wandb_logger.watch(model, 
                log='gradients', 
                log_freq=10)

        # wandb_logger = None

        hyperparams = {'gpus':[1], 
                'max_epochs': 25, 
                'batch_size': model.encoder._batch_size,
                'accumulate_grad_batches': 80,
                'learning_rate': model.encoder._learning_rate,
                'feature_dimension': model.encoder._feature_dimension,
                'model_dimension': model.encoder._model_dimension,
                'seq_len': model.encoder._seqlen,
                'num_transformer_layers': model.encoder._num_layers,
                'num_transformer_layers': model.encoder._num_heads,
                'optimizer': 'Adam',
                'scheduler': 'n/a',}

        trainer = pl.Trainer(
                default_root_dir='/home/sgurram/Desktop/video_lava', 
                gpus=[1], 
                max_epochs=100, 
                accumulate_grad_batches=80,
                logger=wandb_logger,
                profiler=True) 
        
        trainer.fit(model)

def train_classifier():
        # wandb_logger = None

        model = EvalLightning(logger=wandb_logger)

        hyperparams = {'gpus':[1], 
                'max_epochs': 25, 
                'batch_size': model.classifier.batch_size,
                'accumulate_grad_batches': 8,
                'learning_rate': model.classifier.learning_rate,
                'feature_dimension': model.classifier.feature_dimension,
                'model_dimension': model.classifier.model_dimension,
                'num_modalities': 3,
                'type_modalities': 'avt', 
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

