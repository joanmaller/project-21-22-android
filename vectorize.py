import os
import sys
import json
import settings

#settings.RESULTS = "results/"
#settings.KNOWN_FEATURES = "known_features.json"
#filename = "data.json"

known_features = set()

def label_file(sample_name):
    good_apks = os.listdir(settings.GOOD_APK_DIR)
    if sample_name+".apk" in good_apks:
        return "goodware"
    else:
        return "malware"


if __name__ == '__main__':

    if not os.path.exists(settings.KNOWN_FEATURES):
        print("\n[E]\tMissing known features file...")
        sys.exit()

    file = open(settings.KNOWN_FEATURES, "r")
    known_features.update(json.load(file))
    file.close()

    data = []
    labels = []

    for f in os.listdir(settings.RESULTS):

        apk_name = f.split("drebin")[0]
        print("[I]\tProcessing ", apk_name)

        file = open(settings.RESULTS+f)
        features = json.load(file).keys()
        file.close()

        tmp_vect = list()

        for known_f in sorted(known_features):
            if known_f in features:
                tmp_vect.append(1)
            else:
                tmp_vect.append(0)

        data.append(tmp_vect)
        lbl = label_file(apk_name)
        
        if "good" in lbl:
            labels.append(0)
        elif "mal" in lbl:
            labels.append(1)
    
    output_file = open(settings.X_DATA, "w")
    json.dump(data, output_file)
    output_file.close()
    print("[I]\tDone! Wrote data to", settings.X_DATA)
  
    output_file = open(settings.Y_LABELS, "w")
    json.dump(labels, output_file)
    output_file.close()
    print("[I]\tDone! Wrote labels to", settings.Y_LABELS)

