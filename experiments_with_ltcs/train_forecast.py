import argparse
import datetime as dt
import os
import optuna

from modules import CheetahData, TrafficData, OccupancyData, NeuronData, NeuronLaserData
from modules import ForecastModel
from modules import MSELossfuture, NMSELoss
import torch
torch.set_float32_matmul_precision('medium')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


data_classes = {
    "cheetah": CheetahData,
    "traffic": TrafficData,
    "occupancy": OccupancyData,
    "neurons": NeuronData,
    "neuronlaser": NeuronLaserData
}

study_names = {
    "cheetah":2024070307,
    "occupancy":20240626131851
}

def optimise_sigma(model,args,trial):
    sigma = trial.suggest_int("sigma",1,8)
    some_data_class = data_classes[args.dataset]
    dataset_data = some_data_class(future=args.future,seq_len=args.seq_len,iterative_forecast=args.iterative_forecast,sigma=sigma)
    val_score = execute_trial(dataset_data,model,args,trial) 
    return val_score

def optimise_lr(model,dataset_data, args,trial):
    args.lr = trial.suggest_float("learning_rate",1e-4,5e-2)
    val_score = execute_trial(dataset_data,model,args,trial) 
    return val_score

def optimise_seq_len(model,args,trial):
    seq_len = trial.suggest_discrete_uniform("seq_len",32,80,16)
    some_data_class = data_classes[args.dataset]
    dataset_data = some_data_class(future=args.future,seq_len=seq_len,iterative_forecast=args.iterative_forecast,sigma=args.sigma)
    val_score = execute_trial(dataset_data,model,args,trial) 
    return val_score

def optimise_model_params(model,dataset_data, args,trial):
    # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
    args.lr = trial.suggest_float("learning_rate",1e-4,5e-2)
    args.cosine_lr = trial.suggest_int("cosine_lr",0,1) if args.cosine_lr else 0
    args.size  = trial.suggest_int("size",model.out_features +3,40,step=4)
    val_score = execute_trial(dataset_data,model,args,trial) 
    return val_score

def optimise_size(model,dataset_data, args,trial):
    args.size  = trial.suggest_int("size",model.out_features +3,40,step=4)
    val_score = execute_trial(dataset_data,model,args,trial) 
    return val_score

def execute_trial(dataset_data,model,args,trial = None):
    if not args.scheduled_sampling:
        val_scores = 0
        for cross_val_fold in range(5):
            val_score = fit_for_fold(dataset_data,model,args,cross_val_fold,trial = trial)
            val_scores += val_score
        return val_scores / 5
    else:
        model.set_data(dataset_data)
        val_score = model.fit(trial=trial,epochs=args.epochs,gpus=args.gpus,learning_rate=args.lr,cosine_lr=args.cosine_lr,future_loss=args.future_loss,
                                            model_type=args.model,mixed_memory=args.mixed_memory,model_size=args.size)
        return val_score

def cross_validate_model(dataset_data,model,args):
    val_scores = 0
    model_id = model.model_id
    for cross_val_fold in range(4,5):
        model.set_model_id(int(f"{model_id}0{cross_val_fold}"),args.checkpoint_id)
        val_score = fit_for_fold(dataset_data,model,args,cross_val_fold)
        # val_scores += val_score[0]
        print("fold score: ", val_score)
    # print("cv score: ", val_scores / 5)

def fit_for_fold(dataset_data,model,args,cross_val_fold,trial=None):
    print(f"--- fold {cross_val_fold} ---")
    dataset_data.load_data(cross_val_fold=cross_val_fold+1)
    model.set_data(dataset_data)
    val_score = model.fit(trial=trial,epochs=args.epochs,gpus=args.gpus,learning_rate=args.lr,cosine_lr=args.cosine_lr,future_loss=args.future_loss,
                                            model_type=args.model,mixed_memory=args.mixed_memory,model_size=args.size,reset=args.reset)
    return val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--size',default=32,type=str)
    parser.add_argument('--mixed_memory',action='store_true')
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--gpus', nargs='+', type=int,default = None)    
    parser.add_argument('--lr',default=0.02,type=str)
    parser.add_argument('--cosine_lr',action='store_true')
    parser.add_argument('--dataset',default="cheetah",type=str) 
    parser.add_argument('--seq_len',default=32,type=str)
    parser.add_argument('--future',default=1,type=int)
    parser.add_argument('--iterative_forecast',action='store_true')
    parser.add_argument('--optimise',action='store_true')
    parser.add_argument('--cv',action='store_true')
    parser.add_argument('--pruning',action='store_true')
    parser.add_argument('--model_id_shift',type=int,default=0)
    parser.add_argument('--future_loss',action='store_true')
    parser.add_argument('--binwidth',default =0.05, type= float)
    parser.add_argument('--model_id',default =0, type= int)
    parser.add_argument('--sigma',default =7, type= str)
    parser.add_argument('--checkpoint_id',default =0, type= int)
    parser.add_argument('--reset',action='store_true')
    parser.add_argument('--scheduled_sampling',action='store_true')
    parser.add_argument('--no_validation_data',action='store_true') # only applies to non-optuna experiments

    args = parser.parse_args()
    use_optuna = (args.sigma == "optimise" or args.optimise or args.seq_len == "optimise" or args.lr == "optimise" or args.size == "optimise")

    assert args.future > 0 , "Future should be > 0"
    some_data_class = data_classes[args.dataset]
    dataset_data = some_data_class(future=args.future,seq_len=args.seq_len if args.seq_len != "optimise" else 48,binwidth=args.binwidth,
        iterative_forecast=args.iterative_forecast,sigma=args.sigma if args.sigma != "optimise" else 7,scheduled_sampling = args.scheduled_sampling)
    dataset_data.use_validation_data = (not args.no_validation_data)
    task = args.dataset + "_forecast"
    checkpoint_id = None
    if not args.model_id:
        model_id = str(int(dt.datetime.today().strftime("%Y%m%d%H"))  + args.model_id_shift)
    else:
        model_id = args.model_id
    if args.checkpoint_id :# we copy checkpoint to new model
        checkpoint_id = args.checkpoint_id

    
    print(f" --------- model id: {model_id} --------- ")
    
    model = ForecastModel(task=task,model_id = model_id,_data=dataset_data, checkpoint_id=checkpoint_id)

    
    if use_optuna:
        storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{task}.db",
                engine_kwargs={"connect_args": {"timeout": 100}},
        )
        study_name= str(int(model_id))
        if args.sigma == "optimise" or args.seq_len == "optimise" or args.size == "optimise":
            pruner = optuna.pruners.NopPruner()
            sampler=optuna.samplers.BruteForceSampler()
            if args.seq_len == "optimise":
                study_name= str(int(model_id))+"seq_len"
        else:
            pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),patience=4) if args.pruning else optuna.pruners.NopPruner()
            sampler = optuna.samplers.TPESampler()

        if args.reset:
            optuna.delete_study(study_name= study_name,storage=storage,)
        study = optuna.create_study(direction="minimize", pruner=pruner,
                        study_name= study_name,sampler=sampler,
                        storage=storage,load_if_exists=True)

        if args.sigma == "optimise":
            study.optimize(lambda trial: optimise_sigma(model,args,trial),
                        n_trials=100)

        elif args.seq_len == "optimise":
            study.optimize(lambda trial: optimise_seq_len(model,args,trial),
                            n_trials=100)

        elif args.size == "optimise":
            study.optimize(lambda trial: optimise_size(model,dataset_data,args,trial),
                            n_trials=20)
            
        elif args.lr == "optimise":
            study.optimize(lambda trial: optimise_lr(model,dataset_data,args,trial),
                n_trials=100)
        else:
            study.optimize(lambda trial: optimise_model_params(model,dataset_data,args,trial),
                n_trials=100)
                
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    elif args.cv:
        cross_validate_model(dataset_data,model,args)
    elif not args.optimise:
        model.fit(epochs=args.epochs,gpus=args.gpus,model_type=args.model,mixed_memory=args.mixed_memory,model_size=args.size, learning_rate=args.lr,cosine_lr=args.cosine_lr,future_loss=args.future_loss,reset = args.reset)
        model.test(iterative_forecast=args.iterative_forecast,checkpoint="last")
        model.test(iterative_forecast=args.iterative_forecast,checkpoint="best")
    