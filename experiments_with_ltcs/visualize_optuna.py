import optuna.visualization as vis
import optuna
import os
import matplotlib.pyplot as plt


# Replace 'sqlite:///your_database.db' with the path to your Optuna database file
files = os.listdir(os.getcwd())
databases = [ fi for fi in files if  fi.endswith(".db") ]
for db_name in databases:
  print(f"\n\n=========={db_name}==========\n")
  storage = optuna.storages.RDBStorage(
      url=f"sqlite:///{db_name}",
      engine_kwargs={"connect_args": {"timeout": 30}},
  )
  
  #study = optuna.load_study(storage=storage,study_name="finetune_macula_bscan")
  
  # Plot optimization history
  print('getting plot')
  plt.figure()
  vis.plot_optimization_history(study)
  plt.savefig(f"optimization_history_{db_name}.png")
  plt.close()
  print('plot closed')
  study_summaries = optuna.get_all_study_summaries(storage=storage)
  for summary in study_summaries:
    print(f"Study {summary.study_name} with {summary.n_trials} trials \nStarted at {summary.datetime_start} \nBest trial: {summary.best_trial}\n----------------------------\n")
  print(study_summaries)
  # Plot parameter importances
  #plt2 = vis.plot_param_importances(study)
  #plt2.show()
  
