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
   
    def predict_for_laser_separately(self,in_x,pos_laser_value,neg_laser_value):
        x_pos = x_neg = in_x # (1,seq_len,neurons)
        y_hat_pos = y_hat_neg  = torch.empty((1,self.n_iterative_forecasts,in_x.shape[-1]), device=self.device)
        for i in range(self.n_iterative_forecasts):
            """we only add a 1 for laser activation at the 2 bins with most lag, no other bins"""
            x_pos_in = torch.cat((x_pos,torch.full((1,32,1),pos_laser_value if i >= (self.n_iterative_forecasts-2) else neg_laser_value,device=self.device)),dim=-1)
            x_neg_in = torch.cat((x_neg,torch.full((1,32,1), neg_laser_value,device=self.device)),dim=-1)
            x = torch.cat((x_pos_in,x_neg_in),dim=0)
            pred, _ = self._model.forward(x) # (2, last timestep, neurons(excluding activation))
            (next_step_pos,next_step_neg) = pred
            y_hat_pos[:,i] = next_step_pos[-1:,:]
            y_hat_neg[:,i] = next_step_neg[-1:,:]
            x_pos = torch.cat((x_pos[:,1:],next_step_pos[-1:,:].unsqueeze(0)),dim=1)
            x_neg = torch.cat((x_neg[:,1:],next_step_neg[-1:,:].unsqueeze(0)),dim=1) 
        """TODO Now we compare the pos (stimulation) predictions and neg (absence) predictions"""
        """Now we plot the recorded history of the signal followed by the forecasts
                for every neuron + activation
                make sure to include the laseractivation feature in the predictions
        """
        y_hat_pos_de = self.denormalize(y_hat_pos.detach().cpu(),"y").flatten(0,1) # (5,17)
        y_hat_neg_de = self.denormalize(y_hat_neg.detach().cpu(),"y").flatten(0,1)
        else :
            y = y[:,-self.future:,:]
            """forecast directly multiple steps ahead with 1 model call"""
            x_pos = x_
            y_hat, _ = self.model.forward(x) # (B_size, Timesteps, Features)
            y_hat = y_hat[:,-self.future:,:] #(sequences,timesteps,neurons) > (sequences,futures,neurons)

            y_de = self.denormalize(y.detach().cpu(),"y").flatten(0,1)
            y_hat_de = self.denormalize(y_hat.detach().cpu(),"y").flatten(0,1)
            error = (y_de - y_hat_de)
            return (y_de,y_hat_de,error)
    
    def predict_with_real_laser_data(self,x,y):
        # from y we take out the futures of each sequence
        # the shape of y is always (nsequences, seq len, features)
        # in case of not iterative we just select as below, resulting in size (16,5,17)
        # in case of iterative we do the same, resulting in (5,1,17)
        if self.iterative_forecast:
            y_hat = torch.zeros((1,y.shape[0],y.shape[2]), device=x.device) #(1, futures, neurons)
            y = y[:,-self.future:,:].transpose(0,1)
            n_predictions = y.shape[1]
            input_x = x[:1]  # first sequence of (\timesteps) length
            activation_status = int(x[0,-1,-1])
            """ forecast by recursive model calls"""
            for i in range(n_predictions):
                next_step, _ = self.model.forward(input_x)  # (1, Timesteps, neurons) 
                # we assume we deal with laser activity data which is part of x but not part of y 
                y_hat[:,i] = next_step[:,-1:,:] 
                next_step_x = torch.cat((next_step[:, -1:, :],torch.full((1,1,1),activation_status if i==0 else 0,device=x.device)),dim=-1)
                input_x = torch.cat((input_x[:, 1:, :], next_step_x), dim=1) #input for next prediction
        else :
            y = y[:,-self.future:,:]
            """forecast directly multiple steps ahead with 1 model call"""
            y_hat, _ = self.model.forward(x) # (B_size, Timesteps, Features)
            y_hat = y_hat[:,-self.future:,:] #(sequences,timesteps,neurons) > (sequences,futures,neurons)

        y_de = self.denormalize(y.detach().cpu(),"y").flatten(0,1)
        y_hat_de = self.denormalize(y_hat.detach().cpu(),"y").flatten(0,1)
        error = (y_de - y_hat_de)
        return (y_de,y_hat_de,error)
    
    def predict_step(self,batch,batch_idx):
        x, y = batch                 # _batch = (B_size, Timesteps, Features) 
        if self.analyse_laser_effect:
            pass
        else:
            self.predict_with_real_laser_data(x,y)


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
        # p = 0.7 - 0.6*(self.current_epoch / self.trainer.max_epochs)  # Linearly decreasing. start at 0.8 and end at 0.1
        p = 0.7 - 0.6*(self.global_step / self.n_iterations)
        x, y = batch
        y_hat = torch.empty_like(y)
        input_x = x[:1]
        if torch.rand(1) < p:
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