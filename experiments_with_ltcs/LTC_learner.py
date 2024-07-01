# LightningModule for training a RNNSequence module
import pytorch_lightning as pl
import torch
import torch.utils.data as data

class SequenceLearner(pl.LightningModule):
    # from: https://github.com/mlech26l/ncps/blob/master/examples/pt_example.py
    def __init__(self, model, loss_func, _loaderfunc,n_iterations,cosine_lr=False,learning_rate=0.005):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.loss_func = loss_func # nn.MSELoss()
        self._loaderfunc = _loaderfunc
        self.n_iterations = n_iterations
        self.cosine_lr = cosine_lr
        self.save_hyperparameters(ignore="_loaderfunc")
    
    def train_dataloader(self):
        # return self.train_dataloader
        return self._loaderfunc(subset="train")

    def val_dataloader(self):
        # return self.val_dataloader
        return self._loaderfunc(subset="valid")
    
    def test_dataloader(self):
        # return self.test_dataloader
        return self._loaderfunc(subset="test")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = self.loss_func(y_hat, y)
        mae = torch.mean( torch.abs(y-y_hat))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = self.loss_func(y_hat, y)
        mae = torch.mean(torch.abs(y-y_hat))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae",mae,prog_bar=True)
        # self.log_dict({
        #     "accuracy":accuracy,
        #     "loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.cosine_lr:
            for param_group in optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = self.lr
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =self.n_iterations ,eta_min=self.lr/20)
            }
        else:
            return optimizer
        