import os
import json

results_dir = "results/"
filename = "known_features.json"

features_set = set()

if __name__ == '__main__':

    #read from file the known features
    if os.path.exists(filename):
        output_file = open(filename, "r")
        features_set.update(json.load(output_file))
        output_file.close()

    for f in os.listdir(results_dir):
        file = open(results_dir+f)
        data = json.load(file)
        file.close()
        
        features_set.update(data.keys())

        output_file = open(filename, "w")
        json.dump(list(features_set), output_file)
        output_file.close()

    print("\n[I]\tFound features: ", len(features_set))
