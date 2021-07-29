
from absl import app, logging, flags
import torch
import pytorch_lightning as ptl

from aai.experimental.sgurram.lava.downstream.esc50.data import ESC50Dataset


class ESC50DownstreamProbe(ptl.LightningModule):

    def __init__(self, args):
        super(ESC50DownstreamProbe, self).__init__()
        self.save_hyperparameters()

        self._num_classes = 50
        self._input_feature_dim = 128
        self._model_dim = 512
        self._lr = 3e-4

        self.model = torch.nn.Sequential(*[
            torch.nn.BatchNorm1d(self._input_feature_dim),
            torch.nn.Linear(self._input_feature_dim, self._model_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self._model_dim),
            torch.nn.Linear(self._model_dim, self._model_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self._model_dim),
            torch.nn.Linear(self._model_dim, self._model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self._model_dim, self._num_classes),
        ])

    def training_step(self, batch, batch_idx):
        self.model.train()
        logits = self.model(batch['features'])
        loss = torch.nn.functional.cross_entropy(logits, batch['class'])
        accuracy = (logits.argmax(dim=-1) == batch['class']).float().mean()

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        logits = self.model(batch['features'])
        loss = torch.nn.functional.cross_entropy(logits, batch['class'])
        accuracy = (logits.argmax(dim=-1) == batch['class']).float().mean()

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self, ):
        return torch.optim.Adam(self.model.parameters(), lr=self._lr)


flags.DEFINE_string('root', default='/data/audio/ESC-50-master/', help='The root of the ESC50 directory')
FLAGS = flags.FLAGS

def main(*unused_argv):
    # Get the datasets
    train_dataloader = torch.utils.data.DataLoader(
        ESC50Dataset(root=FLAGS.root, split='train'), batch_size=128, shuffle=True, num_workers=4)
    eval_dataloader = torch.utils.data.DataLoader(
        ESC50Dataset(root=FLAGS.root, split='eval'), batch_size=128, shuffle=False, num_workers=4)

    model = ESC50DownstreamProbe(None)

    # Get the trainer
    trainer = ptl.Trainer(
        gpus=1,
        logger=ptl.loggers.WandbLogger(project='lava-esc50')
    )
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    app.run(main)
