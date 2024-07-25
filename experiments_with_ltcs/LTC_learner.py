# LightningModule for training a RNNSequence module
import pytorch_lightning as pl
import torch
import torch.utils.data as data

class SequenceLearner(pl.LightningModule):
    # from: https://github.com/mlech26l/ncps/blob/master/examples/pt_example.py
    def __init__(self, model =None, loss_func =None, _loaderfunc=None,n_iterations=None,iterative_forecast=False,cosine_lr=False,learning_rate=0.005):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.loss_func = loss_func # nn.MSELoss()
        self._loaderfunc = _loaderfunc
        self.n_iterations = n_iterations
        self.cosine_lr = cosine_lr
        self.iterative_forecast =iterative_forecast
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
        pass    

    def predict(self,dataloader,iterative_forecast=True):
        if iterative_forecast:
            in_x, in_x_de = dataloader(subset="predict").x.permute(1,0,2) # in shape of (seq_len,1,neurons) > (1,seq_len,neurons)
            in_x = in_x[:,-32:,:] # fetch last 32 bins
            """Here we add the recordings to the history"""
            recorded_history = torch.concat((recorded_history[:32],in_x[0]), dim= 0) # 
            
            x_pos,x_neg = in_x
            x_pos[:,-1,-1] = 1
            x_neg[:,-1,-1] = 0
            x = torch.cat(x_pos,x_neg,dim=0)
            if iterative_forecast:    
                n_predictions = self.n_iterative_forecasts
                y_hat_pos = y_hat_neg  = torch.zeros((1,self.n_iterative_forecasts,in_x.shape[-1]), device=x.device)
                for i in range(n_predictions):
                    (next_step_pos,next_step_neg), _ = self._model.forward(x)  # (2, Timesteps, neurons)
                    print(next_step_pos.shape) # check that it is 3 -dim
                    """we only add a 1 at the first following and not the other bins!"""
                    x_pos = torch.cat((x_pos[:,1:,:],torch.full((1,1,1),1 if i ==0 else 0)),dim=-1)
                    x_neg = torch.cat((x_neg[:,1:,:],torch.full((1,1,1),0)),dim=-1) 
                    # include the laser activation in the forecast
                    y_hat_pos[:,i,:] = x_pos[:,-1,:]
                    y_hat_neg[:,i,:] = x_neg[:,-1,:]
                    x = torch.cat((x_pos,x_neg),dim=0)

            else :
                (y_hat_pos, y_hat_neg), _ =self._model.forward(x)
                """TODO include the laser activation at the first future position (-future)!"""
                y_hat_pos = y_hat_pos[:,-self.future,:]
                y_hat_neg = y_hat_neg[:,-self.future,:]
            y_hat_pos_de = self.denormalize(y_hat_pos,"y")
            y_hat_neg_de = self.denormalize(y_hat_neg,"y")


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
        