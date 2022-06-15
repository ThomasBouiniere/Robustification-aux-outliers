# # main file

# roadmap
# 1.1 from here you can launch the trained algorithms and visualize the results
# NB algorithms are trained and saved on their own modules
# 1.2: exporting a usable criterion for outliers
# 1.3: training of new models on the go with function train_model ...
# 1.4: interface ?

# imports
from models.NAE_remi import Net as NAE_remi
from models.AE_tim import Autoencoder as AE_tim
from models.AE_ThomasdMdP import Autoencoder as AE_ThomasdMdP
from models.AE_thomasB import Autoencoder as AE_thomasB
from models.AE_leo import Autoencoder as AE_leo
import torch
from visu import visu

classes = dict()
classes['AE_leo'] = AE_leo
classes['AE_thomasB'] = AE_thomasB
classes['AE_ThomasdMdP'] = AE_ThomasdMdP
classes['AE_tim'] = AE_tim
classes['NAE_remi'] = NAE_remi
classes['NAE_gpu'] = NAE_remi
classes['AE_remi'] = NAE_remi
# etc for other models


# global variables to load models onto
Net = None
Autoencoder = None


def get_data(params: dict):
    # get the dataloader (PyTorch object) from data.py
    # params : test/train/validation sets and how to slice them ; random=42 etc, batches of number, lengths ...
    # TODO implement it !
    # TODO try and except error to see if data is adequate
    # NB should be independant from the datatype ? like if we use timeseries later
    # the only change should be to data.py

    # ajouter un paramètre pour savoir si on veut train ou test
    from data import data
    return data(params)


def get_models(params: dict):
    '''
    :model_name: nomenclature of model ex: AE_leo-17-0001.pth
    '''
    global Autoencoder, Net
    # for roc, get the saved model from saved_models (file in .pth
    # for tab, get the 10 saved models with each digit as an outlier
    # refer to README.md for nomenclature of the name

    model_name = params['model_name'][0].split('-')[0]
    # if model_name == 'NAE_remi':
    #     exec('from models.{} import Net'.format(model_name), globals())
    #     print(1)
    # else:
    #     exec('from models.{} import Autoencoder'.format(model_name), globals())

    if model_name not in classes.keys():
        raise Exception(
            f"The name {model_name} is not added to the classes dict -- see def in main.py")
    # remi to be changed to gpu
    try:
        if ('remi' in model_name) or ('gpu' in model_name):
            Net = classes[model_name]
        else:
            Autoencoder = classes[model_name]
    except:
        raise Exception(f"The {model_name} class not found.")

    models = []
    print(params['model_name'])

    for model in params['model_name']:
        try:
            print(["saved_models/{}".format(model)])

            if torch.cuda.is_available():
                model_loaded = torch.load("saved_models/{}".format(model))
            else:
                model_loaded = torch.load(
                    "saved_models/{}".format(model), map_location='cpu')

            models.append(model_loaded)
        except:
            print('Fail to load')
            raise Exception(f"Failed to load {model_name}.")
    return models


def visualize(params: dict, dataloader, models):
    # use the choice of visualisation for the model(s) selected and the data selected as true/false
    # params dico : model name for plotting, what metrics to compute, plot mcmc etc ?
    # TODO : implement for 1 model
    # TODO : for more than 1 models
    # TODO : getting out a criterion function automatically + uncertainty on it based on results
    token = visu(params, dataloader, models)
    # TODO check results
    # TODO visu returns a criterion (boolean classifier) to use
    return token


def main(params: dict):
    # sequence of action
    loader = get_data(params)
    print("data loaded")
    models = get_models(params)
    print("model loaded")
    results = visualize(params, loader, models)
    print("all done")
    return results


if __name__ == "__main__":
    # construct dict with console
    # what to put in dict ?
    # refer to each function in main
    params = {}  # TODO a file to make sure we all have the same ?
    visu_choice = input("choice of visualisation : roc or tab")
    params["dataset"] = "test"  # "train" for training
    params['visu_choice'] = visu_choice
    liste_outliers = input('outliers').split(',')
    for i in range(len(liste_outliers)):
        liste_outliers[i] = int(liste_outliers[i])
    params['outliers'] = liste_outliers

    if visu_choice == "roc":
        model_name = input("model_name")

    if visu_choice == "tab":

        model_name = input('models_name')
    params["model_name"] = model_name.split(',')

    main(params)
