# Setting the parameters for anonymization
k_value = 45          # K-anonymity
l_value = 2           # L-closeness
t_value = 0           # T-diversity
epsilon_value = 0.5   
delta_value = 0.001   

# Loading the Dataset
import numpy as np
import pandas as pd

# Loading the dataset into memory
dataset_path = "data_sample.csv" 
df = pd.read_csv(dataset_path, sep=",", engine="python")

print("Total number of rows in the dataset:", len(df))

# Check the dataset parsing
print(df.head())
check_dataset = input("\nQ> Is the dataset properly parsed? (y/n): ")
if check_dataset.lower() == "n":
    print("[Error: Dataset parsing is incorrect!")
    exit(0)

# Preprocessing the data
df.dropna(axis=0, inplace=True)
print("After pre-processing, the total number of rows is:", len(df))

# Collecting the sttribute details
attributes = dict()
for col in df.columns:
    print("\nAttribute: '%s'" % col)
    while True:
        print("\n\t1. Identifier\n\t2. Quasi-identifier\n\t3. Sensitive\n\t4. Insensitive")
        choice = int(input("Q> Please select the attribute type: "))
        if choice not in [1, 2, 3, 4]:
            print("[Error] Please enter a value 1, 2, 3, or 4!")
        else:
            break
    attributes[col] = {'dataType': df[col].dtype, 'attributeType': ["Identifier", "Quasi-identifier", "Sensitive", "Insensitive"][choice-1]}
    if df[col].dtype.name == "object":
        df[col] = df[col].astype("category")

# For DP stats calculation, a copy of the dataset is made
orig_df = df.copy()

qi_index = [list(orig_df.columns).index(attr) for attr in attributes if attributes[attr]['attributeType'] == "Quasi-identifier"]
feature_columns = [attr for attr in attributes if attributes[attr]['attributeType'] == "Quasi-identifier"]
sensitive_column = [attr for attr in attributes if attributes[attr]['attributeType'] == "Sensitive"]

# Anonymization
for attribute in attributes:
    if attributes[attribute]['attributeType'] == "Identifier":
        df[attribute] = '*'

# Generalizing quasi-identifiers with k-anonymity
from algorithms.anonymizer import Anonymizer

# Check for any quasi-identifiers
quasi = any(attributes[attr]['attributeType'] == "Quasi-identifier" for attr in attributes)

assert quasi, "Quasi-identifier not found! At least 1 quasi-identifier is required."

anon = Anonymizer(df, attributes)
anonymized_df = anon.anonymize(k_value, l_value, t_value)


# Utility Measures
from utility.DiscernMetric import DM
from utility.CavgMetric import CAVG
from utility.GenILossMetric import GenILoss

print("\n--------- Utility Metrics ---------\n")

raw_dm = DM(orig_df, qi_index, k_value)
raw_dm_score = raw_dm.compute_score()

anon_dm = DM(anonymized_df, qi_index, k_value)
anon_dm_score = anon_dm.compute_score()

print(f"DM score (lower is better): \n  BEFORE: {raw_dm_score} || AFTER: {anon_dm_score} || {raw_dm_score > anon_dm_score}")

# Average Equivalence Class
cavg_raw = CAVG(orig_df, qi_index, k_value)
cavg_raw_score = cavg_raw.compute_score()

cavg_anon = CAVG(anonymized_df, qi_index, k_value)
cavg_anon_score = cavg_anon.compute_score()

import math
print(f"CAVG SCORE (near 1 is better): \n  BEFORE: {cavg_raw_score:.3f} || AFTER: {cavg_anon_score:.3f} || {math.fabs(1-cavg_raw_score) > math.fabs(1-cavg_anon_score)}")

# Loss metric gen i
geniloss = GenILoss(orig_df, feature_columns)
geniloss_score = geniloss.calculate(anonymized_df)

print(f"GenILoss: [0: No transformation, 1: Full suppression] \n Value: {geniloss_score}")

# Exporting the data
export_path = "AnonymizedData"
export_choice = input("Do you want to export the anonymized dataset (y/n): ")
if export_choice.lower() != 'y':
    exit(0)

print("\nExporting the anonymized dataset........ ")

# using XlsxWriter as the engine for createing a pandas excel writer object.
writer = pd.ExcelWriter(export_path + '.xlsx')

qi_index = [list(orig_df.columns).index(attr) for attr in attributes if attributes[attr]['attributeType'] == "Quasi-identifier"]

def paint_bg(v, color):
    return [f"background-color: {color[0]};" for i in v]

# Hide the index by setting it to a new column
anonymized_df = anonymized_df.reset_index(drop=True)

# Apply styling to the Styler object with background colors
anonymized_df = anonymized_df.style.apply(paint_bg, color=['gainsboro', 'ivory'], axis=1)

# Writing the anonymized dataframes to the Excel worksheet.
anonymized_df.to_excel(writer, sheet_name='Data', index=False)

writer._save()






