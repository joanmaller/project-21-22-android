import os
import sys
import json
import settings

results_dir = "results/"
feature_file = "known_features.json"
filename = "data.json"

known_features = set()

def label_file(sample_name):
    good_apks = os.listdir(settings.GOOD_APK_DIR)
    if sample_name+".apk" in good_apks:
        return "goodware"
    else:
        return "malware"


if __name__ == '__main__':

    if not os.path.exists(feature_file):
        print("\n[E]\tMissing known features file...")
        sys.exit()

    file = open(feature_file, "r")
    known_features.update(json.load(file))
    file.close()

    output = list()
    data = []
    labels = []

    for f in os.listdir(results_dir):

        apk_name = f.split("drebin")[0]
        print("[I]\tProcessing ", apk_name)

        file = open(results_dir+f)
        features = json.load(file).keys()
        file.close()

        tmp_vect = list()

        for known_f in known_features:
            if known_f in features:
                tmp_vect.append(1)
            else:
                tmp_vect.append(0)

        i =  label_file(apk_name)
        data_sample = {"Y": label_file(apk_name), "X": tmp_vect}
        output.append(data_sample)
        
        data.append(tmp_vect)
        if "good" in i:
            labels.append(1)
        elif "mal" in i:
            labels.append(0)
  

    output_file = open(filename, "w")
    json.dump(output, output_file)
    output_file.close()
    print("[I]\tDone! Wrote results to", filename)

    
    output_file = open("data_X.json", "w")
    json.dump(data, output_file)
    output_file.close()
    print("[I]\tDone! Wrote data to", "data_X.json")
  
    output_file = open("labels_y.json", "w")
    json.dump(labels, output_file)
    output_file.close()
    print("[I]\tDone! Wrote labels to", "labels_y.json")




