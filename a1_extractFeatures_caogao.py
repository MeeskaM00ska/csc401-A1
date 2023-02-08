#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import re
import csv
import string
import os

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}



cdf = 0

if not cdf:
    BGL_path = "/Users/observer/Desktop/401/code/BristolNorms+GilhoolyLogie.csv"
    WAR_path = "/Users/observer/Desktop/401/code/Ratings_Warriner_et_al.csv"
    alt_path = "/Users/observer/Desktop/401/feats/Alt_IDs.txt"
    left_path = "/Users/observer/Desktop/401/feats/Left_IDs.txt"
    right_path = "/Users/observer/Desktop/401/feats/Right_IDs.txt"
    center_path = "/Users/observer/Desktop/401/feats/Center_IDs.txt"
    alt_array = "/Users/observer/Desktop/401/feats/Alt_feats.dat.npy"
    left_array = "/Users/observer/Desktop/401/feats/Left_feats.dat.npy"
    right_array = "/Users/observer/Desktop/401/feats/Right_feats.dat.npy"
    center_array = "/Users/observer/Desktop/401/feats/Center_feats.dat.npy"

bngl_path = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
Warr_path = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
LEFT_path = "/u/cs401/A1/feats/Left_IDs.txt"
CENTER_path = "/u/cs401/A1/feats/Center_IDs.txt"
RIGHT_path = "/u/cs401/A1/feats/Right_IDs.txt"
right_array_path = "/u/cs401/A1/feats/Right_feats.dat.npy"
ALT_path = "/u/cs401/A1/feats/Alt_IDs.txt"
alt_array_path = "/u/cs401/A1/feats/Alt_feats.dat.npy"


BGL_dict = {}
with open(BGL_path, "r") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for r in reader:
        if "WORD" in r:
            BGL_dict[r["WORD"]] = [
                r['AoA (100-700)'],
                r['IMG'],
                r['FAM']
            ]


Warrigner_dict = {}
with open(WAR_path, "r") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=",")
    for r in reader:
        if "Word" in r:
            Warrigner_dict[r["Word"]] = [
                r['V.Mean.Sum'],
                r['A.Mean.Sum'],
                r['D.Mean.Sum']
            ]


leftArr = np.load(left_array)
left_dict = {}
left_info = open(left_path, "r").read().split()
for r in range(len(left_info)):
    left_dict[left_info[r]] = r


centerArr = np.load(center_array)
center_dict = {}
center_info = open(center_path, "r").read().split()
for r in range(len(center_info)):
    center_dict[center_info[r]] = r


rightArr = np.load(right_array)
right_dict = {}
right_info = open(right_path, "r").read().split()
for r in range(len(right_info)):
    right_dict[right_info[r]] = r


altArr = np.load(alt_array)
alt_dict = {}
alt_date = open(alt_path, "r").read().split()
for r in range(len(alt_date)):
    alt_dict[alt_date[r]] = r


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros(29)
    # TODO: Extract features that rely on capitalization.
    upper_dic = {}
    upper_num = 0
    for token in comment.split():
        w = token.split("/")[0]
        if w.isupper() and len(w) >= 3:
            upper_dic[w] = w.lower()
            upper_num += 1
    feats[0] = upper_num
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    for k in upper_dic.keys():
        lower_word = upper_dic[k]
        comment = comment.replace(k, lower_word)

    # TODO: Extract features that do not rely on capitalization.
    total_comment = re.compile("(\S+)/(?=\S+)").findall(comment)
    for t in total_comment:
        # 2. Number of first-person pronouns
        if t in FIRST_PERSON_PRONOUNS:
            feats[1] += 1
        # 3. Number of second-person pronouns
        elif t in SECOND_PERSON_PRONOUNS:
            feats[2] += 1
        # 4. Number of third-person pronouns
        elif t in THIRD_PERSON_PRONOUNS:
            feats[3] += 1
        elif t == ",":
            feats[7] += 1
        # 14. Number of slang acronyms
        elif t in SLANG:
            feats[13] += 1

    total_tags = re.compile("(?<=\S)/(\S+)").findall(comment)
    # 5. Number of coordinating conjunctions
    total_cc = total_tags.count('CC')
    feats[4] = total_cc
    # 6. Number of past-tense verbs
    total_vbd = total_tags.count('VBD')
    feats[5] = total_vbd
    # 7. Number of future-tense verbs
    feats[6] += len(re.compile(r"going/VBG to/TO [\w]+/VB").findall(comment))
    feats[6] += len(re.compile(r'\b(' + r'|'.join(['\'ll', 'will', 'gonna']) +
                               r')\b').findall(comment))
    # 8. Number of commas
    # TODO: COMMAS

    # 9. Number of multi-character punctuation tokens
    # TODO: change
    feats[8] = len(re.findall(' \W{2,}/', comment))
    # 10. Number of common nouns
    n1 = total_tags.count('NN')
    n2 = total_tags.count('NNS')
    feats[9] = n1 + n2
    # 11. Number of proper nouns
    n1 = total_tags.count('NNP')
    n2 = total_tags.count('NNPS')
    feats[10] = n1 + n2
    # 12. Number of adverbs
    a1 = total_tags.count('RBS')
    a2 = total_tags.count('RB')
    a3 = total_tags.count('RBR')
    feats[11] = a1 + a2 + a3
    # 13. Number of wh- words
    feats[12] = total_tags.count('WDT') + total_tags.count('WP') + total_tags \
        .count('WRB') + total_tags.count('WP$')
    # 14. Number of slang acronyms
    # 15. Average length of sentences, in tokens
    # TODO
    if comment.count('\n') == 0:
        feats[14] = 0
    else:
        feats[14] = len(total_comment) / comment.count('\n')
    # 16. Average length of tokens,
    # excluding punctuation-only tokens, in characters
    # TODO
    token_num = 0
    leng = 0
    for t in total_comment:
        if t not in set(string.punctuation):
            leng += len(t)
            token_num += 1
    if total_comment == "":
        feats[15] = 0
    else:
        if token_num == 0:
            feats[15] = 0
        else:
            feats[15] = leng / token_num
    # 17. Number of sentences.
    feats[16] = comment.count('\n')
    # TODO ------------------------------
    AoA_s = []
    IMG_s = []
    FAM_s = []
    V_s, A_s, D_s = [], [], []
    for t in total_comment:
        if t in BGL_dict:
            AoA, IMG, FAM = \
                [float((val if val != "" else 0)) for val in BGL_dict[t]]
            AoA_s.append(AoA)
            IMG_s.append(IMG)
            FAM_s.append(FAM)
        if t in Warrigner_dict:
            V, A, D = \
                [float((val if val != "" else 0)) for val in Warrigner_dict[t]]
            V_s.append(V)
            A_s.append(A)
            D_s.append(D)
    AoA_s = np.array(AoA_s)
    if len(AoA_s) != 0:
        # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
        feats[17] = np.mean(AoA_s)
    if len(IMG_s) != 0:
        # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
        feats[18] = np.mean(IMG_s)
    if len(FAM_s) != 0:
        # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
        feats[19] = np.mean(FAM_s)
    if len(AoA_s) > 0:
        # 21. Standard deviation of AoA (100-700)
        # from Bristol, Gilhooly, and Logie norms
        feats[20] = np.std(AoA_s)
    if len(IMG_s) > 0:
        # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
        feats[21] = np.std(IMG_s)
    if len(FAM_s) > 0:
        # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
        feats[22] = np.std(FAM_s)
    if len(V_s) != 0:
        # 24. Average of V.Mean.Sum from Warringer norms
        feats[23] = np.mean(V_s)
    if len(A_s) != 0:
        # 25. Average of A.Mean.Sum from Warringer norms
        feats[24] = np.mean(A_s)
    if len(D_s) != 0:
        # 26. Average of D.Mean.Sum from Warringer norms
        feats[25] = np.mean(D_s)
    if len(V_s) > 0:
        # 27. Standard deviation of V.Mean.Sum from Warringer norms
        feats[26] = np.std(V_s)
    if len(A_s) > 0:
        # 28. Standard deviation of A.Mean.Sum from Warringer norms
        feats[27] = np.std(A_s)
    if len(D_s) > 0:
        # 29. Standard deviation of D.Mean.Sum from Warringer norms
        feats[28] = np.std(D_s)

    return feats


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    alt_item = altArr[alt_dict[comment_id]]
    center_item = centerArr[center_dict[comment_id]]
    left_item = leftArr[left_dict[comment_id]]
    right_item = rightArr[right_dict[comment_id]]
    if comment_class == "Alt":
        feat = np.append(feat[:29], alt_item)
    elif comment_class == "Center":
        feat = np.append(feat[:29], center_item)
    elif comment_class == "Left":
        feat = np.append(feat[:29], left_item)
    elif comment_class == "Right":
        feat = np.append(feat[:29], right_item)
    return feat


def main(args):
    # Declare necessary global variables here.
    class_as_int = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}
    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))


    # TODO: Call extract1 for each datatpoint to find the first 29 features.
    # Add these to feats.

    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    # TODO __________________________________________________________
    #for i, comment in enumerate(data):
    #    comment_class = comment["cat"]
    #    feat = extract1(comment["body"])
    #    feats[i][:29] = feat[:29]
    #    feats[i][29:173] = extract2(feat, comment_class, comment["id"])[29:173]
    #    feats[i][173] = class_as_int.index(comment_class)

    for i in range(len(data)):
        # Use extract1 to find the first 29 features for each data point. Add these to feats.
        feats[i][:29] = extract1(data[i]["body"])

        # Use extract2 to copy LIWC features (features 30-173) into feats. (Note that these rely on each
        # data point's class, which is why we can't add them in extract1).
        feats[i][29:173] = extract2(feats[i], data[i]["cat"], data[i]["id"])
        feats[i][173] = class_as_int[data[i]["cat"]]
    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output",
                        help="Directs the output to a filename of your choice",
                        required=True)
    parser.add_argument("-i", "--input",
                        help="The input JSON file, preprocessed as in Task 1",
                        required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
