from absl import app, logging, flags
import torch
import pytorch_lightning as ptl
from aai.experimental.sgurram.lava.downstream.ucf101.data import UCF101Dataset
from aai.experimental.ilianherzi.augmented_video_learning.models.utils import create_fc
from aai.experimental.sgurram.lava.src.utils import visualize_batch_downstream

class UCF101DownstreamProbe(ptl.LightningModule):
    def __init__(
        self, 
        input_channels=128,
        linear_dim=128, 
        learning_rate=3e-4,
        milestones=[50, 100, 150]): 

        super(UCF101DownstreamProbe, self).__init__()
        self.save_hyperparameters()
        
        self.num_classes = 101
        self.input_channels = input_channels
        self.linear_dim = linear_dim
        self.learning_rate=learning_rate
        self.milestones = milestones
        self.model = create_fc(
            self.input_channels, 
            fc_dim_sizes=[self.linear_dim], 
            num_classes = self.num_classes
        )
    
    def _accuracy_score(self, Y_pred, Y):
        return (Y_pred.argmax(dim=-1) == Y).float().mean()

    def training_step(self, batch, batch_index):
        X, Y = batch
        Y_pred = self.model(X)
        loss = torch.nn.functional.cross_entropy(Y_pred, Y)
        accuracy = self._accuracy_score(Y_pred, Y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)

        if X.shape[-1] == 2048:
            a,v = torch.split(X.reshape(X.shape[0], 2, -1), split_size_or_sections=1, dim=1)
            a = a[:32].squeeze()
            v = v[:32].squeeze()
            av_sim = a @ v.T
            if batch_index % 100 == 0:
                visualize_batch_downstream(similarity=av_sim.detach(), prefix='train', dataset='ucf')
        return loss

    def validation_step(self, batch, batch_index):
        X, Y = batch
        Y_pred = self.model(X)
        loss = torch.nn.functional.cross_entropy(Y_pred, Y)
        accuracy = self._accuracy_score(Y_pred, Y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=.1, milestones=self.milestones),
                'interval': 'epoch',
                'frequency': 1,
                'name': 'learning_rate'
            }
        return [optimizer], [scheduler]

flags.DEFINE_string('train_split_path', default='/big/iherzi/ucfTrainTestlist/trainlist01.txt', help=' txt path')
flags.DEFINE_string('val_split_path', default='/big/iherzi/ucfTrainTestlist/testlist01.txt', help=' txt path')

flags.DEFINE_integer('split_id', default=1, help='the id of the ucf101 split')
FLAGS = flags.FLAGS
def main(*args):
    dataset = UCF101Dataset(FLAGS.train_split_path, loader='npy')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)

    v_dataset = UCF101Dataset(FLAGS.val_split_path, loader='npy')
    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=128, num_workers=8, shuffle=False)

    trainer = ptl.Trainer(
        gpus=1,
        # logger=ptl.loggers.WandbLogger(project='lava-ucf101'),
        logger=None
    )
    probe = UCF101DownstreamProbe(2*1024, 512)
    trainer.fit(probe, dataloader, v_dataloader)

if __name__ == '__main__':
    app.run(main)