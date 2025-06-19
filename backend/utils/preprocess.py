def preprocess_input(data):
    order = ['Cement', 'Blast', 'Fly', 'Water', 'Superplasticizer', 'Coarse', 'Fine Aggregate', 'Age']
    return [data[feature] for feature in order]
