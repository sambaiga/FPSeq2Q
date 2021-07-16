
import os
import numpy as np
import optuna
from optuna.study import StudyDirection
from optuna.samplers import TPESampler




def get_search_params(trial, params):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.


    N = {'N': trial.suggest_int("N", 9, 99)}
    params.update(N)

    num_cosines = {'num_cosines': trial.suggest_categorical("num_cosines", [16, 32, 64, 128, 256, 512])}
    params.update(num_cosines)

    #optimizer_name = {'optimizer_name': trial.suggest_categorical("optimizer_name", ["Adam", "AdamW", "SGD"])}
    #params.update(optimizer_name)

    #beta1 = {'beta1': trial.suggest_float("beta1",  0.9, 0.99)}
    #params.update(beta1)


    #momentum = {'momentum': trial.suggest_float("momentum",  0.1, 0.99)}
    #params.update(momentum)

    latent_size = {'latent_size': trial.suggest_categorical("latent_size", [32, 64, 128, 256, 512])}
    params.update(latent_size)

    depth = {'depth': trial.suggest_categorical("depth", [2,3, 4, 5])}
    params.update(depth)

    #batch_size = {'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128])}
    #params.update(batch_size)
    
    #patch_size = {'patch_size': trial.suggest_categorical("patch_size", [2, 4, 8])}
    #params.update(patch_size)

    #expansion_factor = {'expansion_factor': trial.suggest_categorical("expansion_factor", [2, 4])}
    #params.update(expansion_factor)

    #num_head = {'num_head': trial.suggest_categorical("num_head", [2, 4, 8])}
    #params.update(num_head)

    #entropy_loss = {'entropy_loss': trial.suggest_categorical("entropy_loss", [True, False])}
    #params.update(entropy_loss)

    #clipping_value= {'clipping_value':trial.suggest_float("clipping_value", 0.01, 5.0)}
    #params.update(clipping_value)

    #dropout = {'dropout':trial.suggest_float("droput", 0.1, 0.5)}
    #params.update(dropout)

    #kappa = {'kappa':trial.suggest_float("kappa", 0.0, 0.9)}
    #params.update(kappa)

    #margin = {'margin':trial.suggest_float("margin", 1e-6, 1e-1, log=True)}
    #params.update(margin)

    #activation  = {'activation':trial.suggest_categorical("activation", [0, 1, 2, 3])}
    #params.update(activation)

   

    #lr = {'lr':trial.suggest_float("lr", 1e-4, 1e-2, log=True)}
    #params.update(lr)

    #calibration_loss = {'calibration_loss': trial.suggest_categorical("calibration_loss", [True, False])}
    #params.update(calibration_loss)

    

    #out_activation  = {'out_activation':trial.suggest_categorical("out_activation", ["relu", "softplus", None,"eluplus" ])}
    #params.update(out_activation)

    return params


class PatientPruner(optuna.pruners.BasePruner):
      def __init__(self, wrapped_pruner, patience: int, min_delta=0.0):
          """
          The objective should improve by at least min_delta in the last
          patience steps
          """
          self.wrapped_pruner = wrapped_pruner
          self._patience = patience
          self._min_delta = min_delta

      def prune(self, study, trial):
          """
          Returns true if this trial has not improved by min_delta
          any time in the last patience steps
          """
          intermediate_values = trial.intermediate_values

          steps = np.asarray(list(intermediate_values.keys()))

          # Do not prune if number of step to determine are insufficient.
          if steps.size < self._patience + 1:
              return False

          steps.sort()
          # This is the score patience steps ago
          steps_before_patience = steps[:-self._patience]
          scores_before_patience = np.asarray(list(intermediate_values[step] for step in steps_before_patience))
          # And these are the scores after that
          steps_after_patience = steps[-self._patience:]
          scores_after_patience = np.asarray(list(intermediate_values[step] for step in steps_after_patience))

          direction = study.direction
          if direction == StudyDirection.MINIMIZE:
              maybe_prune = np.min(scores_before_patience) - self._min_delta < np.min(scores_after_patience)
          elif direction == StudyDirection.MAXIMIZE:
              maybe_prune = np.max(scores_before_patience) + self._min_delta > np.min(scores_after_patience)
          if maybe_prune:
              # KN: we need to return wrapped pruner's prune value to perform pruning
              return self.wrapped_pruner.prune(study, trial)
          else:
              return False


def run_study(objective, num_trials=2, seed=100, study_name = "Hyper-params" ):
    #Activate the pruning feature. `MedianPruner` stops unpromising 
    # Set up the median stopping rule as the pruning condition
    
    #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
     # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    wrapped_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=10)
    #wrapped_pruner = optuna.pruners.SuccessiveHalvingPruner()
    pruner = PatientPruner(wrapped_pruner, patience=5)
    study = optuna.create_study(sampler = TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True), 
                                pruner=pruner, direction='minimize', 
                                study_name=study_name, 
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(objective, n_trials=num_trials)
    
    print("Number of finished trials: {}".format(len(study.trials)))
    print(f"Best trial: {study.best_trial.number} ")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return study



def get_best_params(default_params, study_name = "Hyper-params-hyper_mlp", seed=777):
    
    if os.path.isfile("{}.db".format(study_name)):
        print('Load best params from optuna study')
        storage_name = "sqlite:///{}.db".format(study_name)
        wrapped_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=10)
        #wrapped_pruner = optuna.pruners.SuccessiveHalvingPruner()
        pruner = PatientPruner(wrapped_pruner, patience=5)
        study = optuna.create_study(sampler = TPESampler(seed=seed), 
                                        pruner=pruner, direction='minimize', 
                                        study_name=study_name, 
                                        storage=storage_name,
                                        load_if_exists=True)
        
        default_params.update(study.best_params)
    return default_params