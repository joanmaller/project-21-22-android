import os
import json
import settings


features_set = set()

if __name__ == '__main__':

    #read from file the known features
    if os.path.exists(settings.KNOWN_FEATURES):
        output_file = open(settings.KNOWN_FEATURES, "r")
        features_set.update(json.load(output_file))
        output_file.close()

    for f in os.listdir(settings.RESULTS):
        file = open(settings.RESULTS+f)
        data = json.load(file)
        file.close()
        
        features_set.update(data.keys())

        output_file = open(settings.KNOWN_FEATURES, "w")
        json.dump(list(features_set), output_file)
        output_file.close()

    print("\n[I]\tFound features: ", len(features_set))
