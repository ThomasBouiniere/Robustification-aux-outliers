# projet_ia_P1.04



# nomenclature des modules pour entraîner les models
AE_X : autoencoder vanilla made by X
naive_X : AE with negative loss for heldout digits made by X
NAE_X : AE with search for negative samples using MCMC made by X

# nomenclature des modèles enregistrés .pth
nom du module
outliers
index - get the info of the model in saved_models/model_info.txt index close to line number is best!

nomenclature : modulename-outliers-index.pth
ex : AE_leo-17-0001.pth

# params KESAKO
"model_name" : str name of the models to load, separated by ','
"dataset" : "train" or "test" est le dataset d'ou tirer les images
"batch_size" : int the length of the dataset batches
"outliers" : list of ints
"models" : list of trained models studied. For a ROC curve, list of one model 
"visu_choice" : str name of the choice of visualization
"use_negative_index" : bool for the naive algorithm

# données à donner en entrée
-Courbe ROC de comparaison de différents outliers : liste des outliers, liste de modèles de la même taille, dans le même ordre
-Courbe ROC de comparaison de différents modèles : un seul outlier, liste des noms des modèles
-Tableau de performance : liste des outliers, liste de modèles de la même taille, dans le même ordre
