import json

def golden_dataset_loader(path):
    with open(path, 'r', encoding='utf-8') as dataset_file:
        data = json.load(dataset_file)
    
    test_cases = data['test_cases']
    return test_cases