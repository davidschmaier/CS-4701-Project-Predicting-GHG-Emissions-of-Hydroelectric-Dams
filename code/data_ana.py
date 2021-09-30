import os, sys, pickle
import numpy as np

data_map = {}
feature_name = []
benifits_impacts = {}

if True or not os.path.exists("all_features.pkl"):
    with open("./new_global_dams_db.csv", "r") as f:
        tot = 0
        for line in f:
            line = line.strip("").split(",")
            if tot == 0:
                for item in line:
                    data_map[item] = []
                    feature_name.append(item)
            else:
                for idx, item in enumerate(line):
                    data_map[feature_name[idx]].append(item) 
            tot += 1
    saved = (feature_name, data_map)
    pickle.dump(saved, open("all_features.pkl", "wb"))
else:
    feature_name, data_map = pickle.load(open("all_features.pkl", "rb"))

print (feature_name)

use_name = "Hydroelectricity"

tot_data = len(data_map[feature_name[0]])
power_sep = {}
other_sep = {}
benifits_impacts[use_name] = []
benifits_impacts["other"] = []

print ("total number of dams: ", tot_data)
power_sep["GEN_GWH_YR"] = []
power_sep["FISH_RATIO"] = []
power_sep["CO2EQ"] = []
other_sep["CAP_MCM"] = []
other_sep["FISH_RATIO"] = []
other_sep["CO2EQ"] = []

power_ratio = {}
other_ratio = {}

power_ratio["FISH_RATIO"] = [] 
power_ratio["CO2EQ"] = [] 
other_ratio["FISH_RATIO"] =  [] 
other_ratio["CO2EQ"] = []



import numbers
power_cnt = 0
other_cnt = 0

for i in range(tot_data):
    if data_map["MAIN_USE"][i] == use_name:
        power_cnt += 1
        try:
            benifits_impacts[use_name].append((float(data_map["GEN_GWH_YR"][i]), float(data_map["RES_BAL"][i]), float(data_map["CO2EQ_total"][i])))
            power_sep["GEN_GWH_YR"].append(float(data_map["GEN_GWH_YR"][i]))
            power_sep["FISH_RATIO"].append(float(data_map["RES_BAL"][i]))
            power_sep["CO2EQ"].append(float(data_map["CO2EQ_total"][i]))
        except:
            continue
        try:
            power_ratio["FISH_RATIO"].append(float(data_map["RES_BAL"][i]) / float(data_map["GEN_GWH_YR"][i]))
            power_ratio["CO2EQ"].append(float(data_map["CO2EQ_total"][i]) / float(data_map["GEN_GWH_YR"][i]))
        except:
            continue
    else:
        other_cnt += 1
        try:
            benifits_impacts["other"].append((float(data_map["CAP_MCM"][i]), float(data_map["RES_BAL"][i]), float(data_map["CO2EQ_total"][i])))
            other_sep["CAP_MCM"].append(float(data_map["CAP_MCM"][i]))
            other_sep["FISH_RATIO"].append(float(data_map["RES_BAL"][i]))
            other_sep["CO2EQ"].append(float(data_map["CO2EQ_total"][i]))
        except:
            continue
        try:
            other_ratio["FISH_RATIO"].append(float(data_map["RES_BAL"][i]) / float(data_map["CAP_MCM"][i]))
            other_ratio["CO2EQ"].append(float(data_map["CO2EQ_total"][i]) / float(data_map["CAP_MCM"][i]))
        except:
            continue
 
print (len(power_sep["GEN_GWH_YR"]), " ", len(power_sep["FISH_RATIO"]), " ", len(power_sep["CO2EQ"]))
print (len(power_ratio["FISH_RATIO"]), " ", len(power_ratio["CO2EQ"]))
cuts = [np.quantile(power_sep["GEN_GWH_YR"], 0.5), np.quantile(power_sep["FISH_RATIO"], 0.5), np.quantile(power_sep["CO2EQ"], 0.5)]
import copy
power_cuts = copy.deepcopy(cuts)
labels = [0 for _ in range(8)]

power_ratio_cuts = [np.quantile(power_ratio["FISH_RATIO"], 0.5), np.quantile(power_ratio["CO2EQ"], 0.5)]

print (power_ratio_cuts)
print (power_ratio["FISH_RATIO"][:20])

def get_ratio_label(a, b, cuts):
    a = 1 if a >= cuts[0] else 0
    b = 0 if b >= cuts[1] else 1
    return (2 * a) + b;

def get_label(a, b, c, cuts):
    a = 1 if a >= cuts[0] else 0
    b = 1 if b >= cuts[1] else 0
    c = 0 if c >= cuts[2] else 1
    return (2 * 2 * a) + (2 * b) + c

print ("total power dam: ", power_cnt)
print ("usable power dam: ", len(power_sep["GEN_GWH_YR"]))
tot = len(benifits_impacts[use_name])
for item in benifits_impacts[use_name]:
    a, b, c = item
    labels[get_label(a, b, c, cuts)] += 1

for i in range(8):
    print ("label: %d, percentage: %.3f" % (i, labels[i] / tot))

cuts = [np.quantile(other_sep["CAP_MCM"], 0.5), np.quantile(other_sep["FISH_RATIO"], 0.5), np.quantile(other_sep["CO2EQ"], 0.5)]
other_ratio_cuts = [np.quantile(other_ratio["FISH_RATIO"], 0.5), np.quantile(power_ratio["CO2EQ"], 0.5)]
other_cuts = copy.deepcopy(cuts)
labels = [0 for _ in range(8)]

print ("total other dam: ", other_cnt)
print ("usable other dam: ", len(other_sep["CO2EQ"]))
tot = len(benifits_impacts['other'])
for item in benifits_impacts['other']:
    a, b, c = item
    labels[get_label(a, b, c, cuts)] += 1

for i in range(8):
    print ("label: %d, percentage: %.3f" % (i, labels[i] / tot))

labels = [0 for _ in range(4)]
for a, b in zip(power_ratio["FISH_RATIO"], power_ratio["CO2EQ"]):
    labels[get_ratio_label(a, b, power_ratio_cuts)] += 1

print ("for ratio power dams")
tot = len(power_ratio["CO2EQ"])
for i in range(4):
    print ("power ratio label: %d, percetange: %.3f" % (i, labels[i] / tot))

labels = [0 for _ in range(4)]
for a, b in zip(other_ratio["FISH_RATIO"], power_ratio["CO2EQ"]):
    labels[get_ratio_label(a, b, other_ratio_cuts)] += 1

print ("for ratio other dams")
tot = len(power_ratio["CO2EQ"])
for i in range(4):
    print ("other ratio label: %d, percetange: %.3f" % (i, labels[i] / tot))

#exit(-1)

# Start generateing datafile for training the power dam

power_dam_features = []
power_ratio_dam_features = []
power_dam_labels = []
power_ratio_dam_labels = []
other_dam_features = []
other_ratio_dam_features = []
other_dam_labels = []
other_ratio_dam_labels = []

power_unsup_ratio_dam_features = []
other_unsup_ratio_dam_features = []

def convert_ratio_label(l):
    return l

def convert_label(l):
    return l

def useful(feature):
    # Label cannot be used in feature
    if feature == "GEN_GWH_YR" or feature == "GEN_GWH_YR" or feature[:4] == "RES_"or feature[:5] == "CO2EQ" or feature == "FISH_RATIO" or feature == "CAP_MCM":
        return 0
    # Seems ussless to me:
    if feature == "GRAND_ID" or feature == "DAM_NAME" or feature == "MAIN_BASIN" or feature == "SUB_BASIN" or feature == "NEAR_CITY" or feature == "COUNTRY" or feature == "SEC_CNTRY" or feature == "YEAR" or feature == "REM_YEAR" or feature[:4] == "USE_" or feature == "MAIN_USE" or feature == "LAKE_CTRL" or feature == "LONG_DD" or feature == "LAT_DD" or feature == "COUNTRY_1":
        return 0
    # debating features
    if feature == "LAT_KEY\n" or feature[:5] == "pred_":
        return 0
    # too many NA
    if feature == "INSCAP_MW":
        return 0
    return 1

def process(feature, value):
    if value == "NA" or value == "":
        return None
    return value

tot = 0
unsup_tot = 0
no_missing = 0
names = []
for i in range(tot_data):
    if data_map["MAIN_USE"][i] != use_name:
        continue
    tot += 1
    tmp_features = []
    na = 0
    names = []
    for idx, feature in enumerate(feature_name):
        if useful(feature):
            #print (feature, " ", data_map[feature][i], " ", data_map["DAM_NAME"][i])
            names.append(feature)
            cur_feature = process(feature, data_map[feature][i])
            if cur_feature == None:
                na = 1
                break
            tmp_features.append(cur_feature)
    if na:
        continue
    else:
        try:
            a, b, c = (float(data_map["GEN_GWH_YR"][i]), float(data_map["RES_BAL"][i]), float(data_map["CO2EQ_total"][i]))
            no_missing += 1
            aa, bb = float(data_map["RES_BAL"][i]) / float(data_map["GEN_GWH_YR"][i]), float(data_map["CO2EQ_total"][i]) / float(data_map["GEN_GWH_YR"][i])
            power_dam_features.append(np.array(tmp_features))
            power_ratio_dam_features.append(np.array(tmp_features))
            power_dam_labels.append(convert_label(get_label(a, b, c, power_cuts)))
            power_ratio_dam_labels.append(convert_ratio_label(get_ratio_label(aa, bb, power_ratio_cuts)))
            power_unsup_ratio_dam_features.append(np.array(tmp_features))
        except:
            unsup_tot += 1
            power_unsup_ratio_dam_features.append(np.array(tmp_features))
            continue
print ("used features: ", names)
print (len(names))
print ("no missing: ", no_missing, "unsup_tot: ", unsup_tot, " total data: ", tot)
print (power_dam_features[0])
print (power_dam_labels)

power_dam_save = (power_dam_features, power_dam_labels)
pickle.dump(power_dam_save, open("test_power_dam_data.pkl", "wb"))
power_ratio_dam_save = (power_ratio_dam_features, power_ratio_dam_labels)
pickle.dump(power_ratio_dam_save, open("test_power_ratio_dam_data.pkl", "wb"))
pickle.dump(power_unsup_ratio_dam_features, open("test_power_ratio_unsup_dam_data.pkl", "wb"))

tot = 0
unsup_tot = 0
no_missing = 0
for i in range(tot_data):
    if data_map["MAIN_USE"][i] == use_name:
        continue
    tot += 1
    tmp_features = []
    names = []
    na = 0
    for idx, feature in enumerate(feature_name):
        if useful(feature):
            cur_feature = process(feature, data_map[feature][i])
            if cur_feature == None:
                na = 1
                break
            tmp_features.append(cur_feature)
    if na:
        continue
    else:
        try:
            a, b, c = (float(data_map["CAP_MCM"][i]), float(data_map["RES_BAL"][i]), float(data_map["CO2EQ_total"][i]))
            aa, bb = float(data_map["RES_BAL"][i]) / float(data_map["CAP_MCM"][i]), float(data_map["CO2EQ_total"][i]) / float(data_map["CAP_MCM"][i])
            other_dam_features.append(np.array(tmp_features))
            other_ratio_dam_features.append(np.array(tmp_features))
            other_dam_labels.append(convert_label(get_label(a, b, c, other_cuts)))
            other_ratio_dam_labels.append(convert_ratio_label(get_ratio_label(aa, bb, other_ratio_cuts)))
            other_unsup_ratio_dam_features.append(np.array(tmp_features))
            no_missing += 1
        except:
            unsup_tot += 1
            other_unsup_ratio_dam_features.append(np.array(tmp_features))
            pass

print ("no missing: ", no_missing, "unsup_tot: ", unsup_tot, "total_data: ", tot)
print (len(other_ratio_dam_features))

other_dam_save = (other_dam_features, other_dam_labels)
pickle.dump(other_dam_save, open("test_other_dam_data.pkl", "wb"))
other_ratio_dam_save = (other_ratio_dam_features, other_ratio_dam_labels)
pickle.dump(other_ratio_dam_save, open("test_other_ratio_dam_data.pkl", "wb"))
pickle.dump(other_unsup_ratio_dam_features, open("test_other_ratio_unsup_dam_data.pkl", "wb"))








        


