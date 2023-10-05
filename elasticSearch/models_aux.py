from constants import MODEL_NAMES
from text_embeddings import save_models


def get_models(src_paths: list, model_names: list = MODEL_NAMES):
    '''
    src_paths: paths to the documents to be inserted into the database
    model_names: names of the models to be used for embedding
    return: dictionary with model names as keys and the models as values
    '''
    models = {}
    if 'infer' in model_names and (not 'ae' in model_names):    # needs AE for embedding
        model_names = model_names + ['ae']
    for model_name in model_names:
        try: # model exists
            model = save_models.load_model(model_name)
            models[model_name] = model
        except: # model does not exist, create and save it
            model = save_models.train_model(model_name, src_paths)
            models[model_name] = model
            save_models.save_model(model, model_name)
    return models