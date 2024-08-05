# LightningModule for training a RNNSequence module
import pytorch_lightning as pl
import torch
import torch.utils.data as data

class SequenceLearner(pl.LightningModule):
    # from: https://github.com/mlech26l/ncps/blob/master/examples/pt_example.py
    # def __init__(self, model =None, loss_func =None, _loaderfunc=None,n_iterations=None,iterative_forecast=False,cosine_lr=False,lr=0.005):
    def __init__(self,**kwargs):
        super().__init__()
        # self.model = model
        # self.lr = lr
        # self.loss_func = loss_func # nn.MSELoss()
        # self._loaderfunc = _loaderfunc
        # self.n_iterations = n_iterations
        # self.cosine_lr = cosine_lr
        # self.iterative_forecast =iterative_forecast
        # self.future = future
        for (k,v) in kwargs.items():
            setattr(self,k,v)
        self.save_hyperparameters(ignore=["model","_loaderfunc","loss_func","denormalize"])   
    

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
        self.log("train_mae", mae, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = self.loss_func(y_hat, y)
        mae = torch.mean(torch.abs(y-y_hat))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae",mae,prog_bar=False)
        # self.log_dict({
        #     "accuracy":accuracy,
        #     "loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
   
    def predict_step(self,batch,batch_idx):
        x, y = batch                 # _batch = (B_size, Timesteps, Features) 
        # from y we take out the futures of each sequence
        # the shape of y is always (nsequences, seq len, features)
        # in case of not iterative we just select as below, resulting in size (16,5,17)
        # in case of iterative we do the same, resulting in (5,1,17)
        if self.iterative_forecast:
            y_hat = torch.zeros((1,y.shape[0],y.shape[2]), device=x.device) #(1, futures, neurons)
            y = y[:,-self.future:,:].transpose(0,1)
            n_predictions = y.shape[1]
            x = x[:1]  # first sequence of (\timesteps) length
            activation_status = int(x[0,-1,-1])
            """ forecast by recursive model calls"""
            for i in range(n_predictions):
                next_step, _ = self.model.forward(x)  # (1, Timesteps, neurons) 
                # we assume we deal with laser activity data which is part of x but not part of y 
                y_hat[:,i] = next_step[:,-1:,:] 
                next_step_x = torch.cat((next_step[:, -1:, :],torch.full((1,1,1),activation_status if i==0 else 0,device=x.device)),dim=-1)
                x = torch.cat((x[:, 1:, :], next_step_x), dim=1) #input for next prediction
        else :
            y = y[:,-self.future:,:]
            """forecast directly multiple steps ahead with 1 model call"""
            y_hat, _ = self.model.forward(x) # (B_size, Timesteps, Features)
            y_hat = y_hat[:,-self.future:,:] #(sequences,timesteps,neurons) > (sequences,futures,neurons)

        y_de = self.denormalize(y.detach().cpu(),"y").flatten(0,1)
        y_hat_de = self.denormalize(y_hat.detach().cpu(),"y").flatten(0,1)
        error = (y_de - y_hat_de)
        return (y_de,y_hat_de,error)

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
        

class ScheduledSamplingSequenceLearner(SequenceLearner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        p = 0.8 - 0.7*(self.current_epoch / self.trainer.max_epochs)  # Linearly decreasing. start at 0.8 and end at 0.1
        x, y = batch
        y_hat = torch.empty_like(y)
        input_x = x[:1]
        if torch.rand(1) > p:
            for i in range(self.n_iterative_forecasts):
                if i > 0:
                    # take the timestep inputs from pred
                    # take the laser inputs from x : from the last ti
                    laser_value = x[i,-1,-1]
                    predicted_x = torch.cat((predicted[:, -1:, :], torch.full((1,1,1),laser_value,device = predicted.device)), dim=2)
                    input_x = torch.cat((input_x[:, 1:, :], predicted_x), dim=1) #input for next prediction
                predicted, _ = self.model.forward(input_x)
                y_hat[i] = predicted[0]
            pred, _ = self.model.forward(x[self.n_iterative_forecasts:])
            y_hat[self.n_iterative_forecasts:] = pred
        else:
            y_hat, _ = self.model.forward(x)
            y_hat = y_hat.view_as(y)
        loss = self.loss_func(y_hat, y)
        mae = torch.mean( torch.abs(y-y_hat))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=False)
        return {"loss": loss}
 
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