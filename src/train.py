import torch
import torchaudio
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import warnings
import glob
from tqdm import tqdm
import wandb
from absl import app, flags

from aai.experimental.sgurram.lava.src.driver import LAVALightning, EvalLightning
from aai.experimental.sgurram.lava.src.references import save_dir
from aai.experimental.sgurram.lava.src.references import lava_weights_path, old_lava_weights_path
from aai.experimental.sgurram.lava.src.utils import compare_models, get_npy_paths
from aai.experimental.sgurram.lava.src.data import*
from aai.experimental.sgurram.lava.src.metrics import cosine_similarity


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

def train_lava(logging=False, train=True):

        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

        hparams = {'gpus':[0, 1], 
                'max_epochs': 100, 
                'auto_lr_find': False,
                'learning_rate': 5e-5,
                'max_lr': 5e-5,
                'min_lr': 5e-6,
                'pretrained_text': False,
                'multiple_video': False,
                'text_type': 'video titles',
                'video augmentation': ['random crop, temporal sampling', 'hflip', 'color jitter'],
                'auto_scale_batch_size': None,
                'batch_size': 32,
                'accumulate_grad_batches': 2,
                'model_dimension': 1024,
                'feature_dimension': 512,
                'seq_len': 256,
                'loss_functions': ['modal_pairwise_nce', 'triplet_centroid_nce'],
                'metrics': ['unimodal_cos_dist', 'modal_pairwise_cos_sim'],
                'num_transformer_heads': 4,
                'num_transformer_layers': 4,
                'model_path': "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210715_130359-2ws983g9/files/lava/2ws983g9/checkpoints/epoch=12-step=46539.ckpt",
                'dropout': 0.0,
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
                'default_root_dir': '/home/sgurram/Desktop/video_lava',
                'gradient_clip_val': 0.5,
                'gradient_clip_algorithm': 'value',
                }

        model = LAVALightning(
                model_dimension=hparams['model_dimension'], 
                feature_dimension=hparams['feature_dimension'],
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
                cosine_steps_period=hparams['cosine_steps_period'],
                multiple_video=hparams['multiple_video'],
                pretrained_text=hparams['pretrained_text'],)
        
        # model.load_from_checkpoint(hparams['model_path'], strict=True)
        # print(model)
        # model.load_state_dict(torch.load(hparams['model_path'], map_location="cuda:0")['state_dict'], strict=True)
        # model.encoder.load_state_dict(torch.load('overfit_lava.pt'), strict=True)

        if logging:
                wandb_logger = WandbLogger(name='run',project='lava')
                wandb_logger.log_hyperparams(hparams)
                wandb_logger.watch(model, 
                        log='gradients', 
                        log_freq=10)
        else:
                wandb_logger = None

        if not train:
                return model

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
                # resume_from_checkpoint=hparams['model_path'],
                precision=hparams['precision'],
                distributed_backend=hparams['distributed_backend'],
                # gradient_clip_val=hparams['gradient_clip_val'],
                # gradient_clip_algorithm=hparams['gradient_clip_algorithm'],
                # limit_train_batches=2,
                # limit_val_batches=2,
                ) 
        
        # trainer.tune(model)

        trainer.fit(model)

def train_classifier(logging=False, train=True):

        hparams = {'gpus': [1], 
                'max_epochs': 25, 
                'num_classes': 700,
                'feature_dimension': 512,
                'model_dimension': 1024,
                'pretrained_text': False,
                'num_modalities': 1,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'model_path': "/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210626_215155-yqwe58z7/files/lava/yqwe58z7/checkpoints/epoch=6-step=12529.ckpt",
                'model_descriptor': 'lava timesformer 1/3 kinetics data, unshuffled',
                'accumulate_grad_batches': 2,
                'overfit_batches': 0,
                'type_modalities': 'av', 
                'modality_fusion': 'concat',
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
                model_path=hparams['model_path'],
                model=LAVALightning,
                pretrained_text=hparams['pretrained_text'],
        )

        if logging:
                wandb_logger = WandbLogger(name='run',project='lava')
                wandb_logger.log_hyperparams(hparams)
                wandb_logger.watch(model, 
                        log='gradients', 
                        log_freq=10)
        else:
                wandb_logger = None

        if not train:
                return model

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

def probe_features():

        model_path = '/home/sgurram/Projects/aai/aai/experimental/sgurram/lava/src/wandb/run-20210427_180730-1bs8nddy/files/lava/1bs8nddy/checkpoints/epoch=16-step=45332.ckpt'


        model = LAVALightning(batch_size=32,
                model_dimension=1024,
                num_heads=4,
                num_layers=4,
                pretrained_text=False,)

        model.load_state_dict(torch.load(model_path, map_location='cuda:1')['state_dict'], strict=True)

        data = LAVAData("val", False)

        a_encoder = model.encoder.a_encoder
        v_encoder = model.encoder.v_encoder
        t_encoder = model.encoder.t_encoder

        a1, v1, t1, _, _ = data[8000]
        a2, v2, t2, _, _ = data[8000]

        t2, _ = data.process_text(text="A video of skiing snow")

        a = torch.stack((a1, a2))
        v = torch.stack((v1, v1))
        t = torch.stack((t1, t2))

        a = a_encoder(a)
        v = v_encoder(v)
        t = t_encoder(t)

        a = torch.nn.functional.normalize(a, p=2, dim=-1)
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        t = torch.nn.functional.normalize(t, p=2, dim=-1)

        # print(a @ a.T)
        # print(v @ v.T)
        # print(t @ t.T)

        # print(a @ v.T)
        print(v @ t.T)
        print(a @ t.T)

        # a1, v1, t1, = a_encoder(a1.unsqueeze(0)), v_encoder(v1.unsqueeze(0)), t_encoder(t1.unsqueeze(0))
        # a2, v2, t2, = a_encoder(a2.unsqueeze(0)), v_encoder(v2.unsqueeze(0)), t_encoder(t2.unsqueeze(0))


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
                probe_features()