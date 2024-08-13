import training_dataset as train
from regression_model import TestNetParameters, ModelParameters, WideNetParameters

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from torch.optim import Adam
from OptimizationParameters import AdamOptimizationParameters
import torch
import path
torch.manual_seed(path.seed)

advisor = 0
run_index = "H1"
model_param = WideNetParameters()
BATCH_SIZE = 256

NUM_TRIALS = 100
MAX_EPOCHS = 500

dataset = train.AdvisorDataset(advisor)
dataloader = train.GenerateDataloader(dataset=dataset, batch_size=BATCH_SIZE, nval=0.15, ntest=0.25)


def merge_config_space(configspace1 : ConfigurationSpace, configspace2 : ConfigurationSpace):
    configspace1.add_hyperparameters(configspace2._hyperparameters.values())
    return configspace1


configspace_model = model_param.get_configuration_space()
configspace_alg = AdamOptimizationParameters.get_configuration_space()
configspace = merge_config_space(configspace_model, configspace_alg)

scenario = Scenario(configspace, deterministic=True, n_trials=NUM_TRIALS, n_workers=1)


def convert_config_to_param(config : Configuration, modelParam : ModelParameters):
    modelParam.set_from_configuration(config)

    # Run optimization
    optParam = AdamOptimizationParameters(
        # INIT_LR=config["INIT_LR"],
        # WEIGHT_DECAY=config["WEIGHT_DECAY"],
        INIT_LR=config["INIT_LR"],
        WEIGHT_DECAY=config["WEIGHT_DECAY"],
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=MAX_EPOCHS)
    return modelParam, optParam


iteration = 0


def train_loop(config : Configuration, seed : int=0) -> float:
    print("=============================")
    print("=============================")
    # Hyperparameters
    global model_param
    layout, optParam = convert_config_to_param(config, model_param)

    model = layout.instantiate_new_model()
    opt = Adam(model.parameters(), lr=optParam.INIT_LR, weight_decay=optParam.WEIGHT_DECAY)
    global dataloader
    train_dataloader = dataloader.trainDataLoader
    val_dataloader = dataloader.valDataLoader

    global iteration
    run_name = str(iteration) + "/" + str(NUM_TRIALS)
    param, val_loss = train.train(model=model, opt=opt, trainDataLoader=train_dataloader, valDataLoader=val_dataloader, epochs=optParam.EPOCHS, run_name=run_name)
    iteration += 1
    return val_loss


# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train_loop)
incumbent = smac.optimize()

print("===========================================")
print("======= DONE! TRAINING INCUMBENT =========")
print("===========================================")
# Train on best parameters
layout, optParam = convert_config_to_param(incumbent, model_param)
model = layout.instantiate_new_model()
opt = Adam(model.parameters(), lr=optParam.INIT_LR, weight_decay=optParam.WEIGHT_DECAY)
train_dataloader = dataloader.trainDataLoader
val_dataloader = dataloader.valDataLoader

train.train(model=model, opt=opt, trainDataLoader=train_dataloader, valDataLoader=val_dataloader,
            epochs=optParam.EPOCHS)

# Save training from best parameters
save_name = train.get_save_name(advisor=advisor, modelName=layout.getName(), algName=optParam.getName(), run_index=str(run_index), extension=".pickle")
train.save_model_and_hp(model=model, hyperparam=layout, batch_size=BATCH_SIZE, name=save_name, advisor=advisor)