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
        self.save_hyperparameters(ignore=["model","_loaderfunc"])   
    
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
    def __init__(self, sampling_probability, **kwargs):
        super().__init__(kwargs)
        self.sampling_probability = sampling_probability

    def training_step(self, batch, batch_idx):
        x, y = batch
        if np.random.rand() < self.sampling_probability:
            y_hat, _ = self.model.forward(x)
            self.y_hat = torch.cat(x[:,1:],y_hat[:,:1],dim=0)
            y_hat = y_hat.view_as(y)
            loss = self.loss_func(y_hat, y)
            mae = torch.mean( torch.abs(y-y_hat))
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_mae", mae, prog_bar=True)
        return {"loss": loss}

        if np.random.rand() < self.sampling_probability:
            # Use model's prediction as the next input
            next_input = model.predict(current_input[np.newaxis, :, :])
        else:
            # Use the actual next value as the next input
            next_input = current_target[t]
        current_input = np.append(current_input[1:], next_input).reshape(-1, 1)
        
        # Train on the current sequence
        model.train_on_batch(current_input[np.newaxis, :, :], current_target[t])



        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = self.loss_func(y_hat, y)
        mae = torch.mean( torch.abs(y-y_hat))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return {"loss": loss}