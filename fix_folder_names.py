import os
import shutil

#parent_path = r"Z:\Analyses\2022-HGR\BAM2FCGR\Isolates"
parent_path = r"Z:\Analyses\2022-HGR\BAM2FCGR\Communities\N8_S99"
substrings1=['_1_3_','_1.0_3_']
substrings2=['_1_1_','_1.0_1_']


keep_for_later1=[]
keep_for_later2=[]
duplicate1=[]
duplicate2=[]

for FCGR in [f for f in os.listdir(parent_path) if f.startswith("FCGR_")]:

    if substrings1[0] in FCGR:
        if FCGR.replace(substrings1[0], "!!!") in keep_for_later1:
            duplicate1.append(FCGR)
        else:
            keep_for_later1.append(FCGR.replace(substrings1[0], "!!!"))
    elif substrings1[1] in FCGR:
        if FCGR.replace(substrings1[1], "!!!") in keep_for_later1:
            duplicate1.append(FCGR)
        else:
            keep_for_later1.append(FCGR.replace(substrings1[1], "!!!"))

    elif substrings2[0] in FCGR:
        if FCGR.replace(substrings2[0], "!!!") in keep_for_later2:
            duplicate2.append(FCGR)
        else:
            keep_for_later2.append(FCGR.replace(substrings2[0], "!!!"))
    elif substrings2[1] in FCGR:
        if FCGR.replace(substrings2[1], "!!!") in keep_for_later2:
            duplicate2.append(FCGR)
        else:
            keep_for_later2.append(FCGR.replace(substrings2[1], "!!!"))

for FCGR in [f for f in os.listdir(parent_path) if f.startswith("FCGR_")]:
    if FCGR in duplicate1:
        if substrings1[1] in FCGR: # make sure to delete the folder with _1_3_ and not the one with _1.0_3_
            FCGR.replace(substrings1[1], substrings1[0])
        # else: # if the folder has the substring _1_3_ then delete it
        print(f"deleting {FCGR}")
        shutil.rmtree(os.path.join(parent_path, FCGR), ignore_errors=True)

    elif FCGR in duplicate2:
        # make sure to delete the folder with _1_1_ and not the one with _1.0_1_
        if substrings2[1] in FCGR:
            FCGR.replace(substrings2[1], substrings2[0])
        # else: # if the folder has the substring _1_1_ then delete it
        print(f"deleting {FCGR}")
        shutil.rmtree(os.path.join(parent_path, FCGR), ignore_errors=True)
    elif substrings1[0] in FCGR:
        # rename the folder to _1.0_3_
        new_name = FCGR.replace(substrings1[0], substrings1[1])
        print(f"renaming {FCGR} to {new_name}")
        # os.rename(os.path.join(root, FCGR), os.path.join(root, new_name))
    elif substrings2[0] in FCGR:
        # rename the folder to _1.0_1_
        new_name = FCGR.replace(substrings2[0], substrings2[1])
        print(f"renaming {FCGR} to {new_name}")
        # os.rename(os.path.join(root, FCGR), os.path.join(root, new_name))

# for root, dirs, files in os.walk(parent_path):
#     # the root folder needs to end with "CGR_l10000_ec10.0"
#     if not root.endswith("CGR_l10000_ec10.0"):
#         continue
#
#     FCGRs = dirs
#     keep_for_later1=[]
#     keep_for_later2=[]
#     duplicate1=[]
#     duplicate2=[]
#     for FCGR in FCGRs:
#         if substrings1[0] in FCGR:
#             if FCGR.replace(substrings1[0], "!!!") in keep_for_later1:
#                 duplicate1.append(FCGR)
#             else:
#                 keep_for_later1.append(FCGR.replace(substrings1[0], "!!!"))
#         elif substrings1[1] in FCGR:
#             if FCGR.replace(substrings1[1], "!!!") in keep_for_later1:
#                 duplicate1.append(FCGR)
#             else:
#                 keep_for_later1.append(FCGR.replace(substrings1[1], "!!!"))
#
#         elif substrings2[0] in FCGR:
#             if FCGR.replace(substrings2[0], "!!!") in keep_for_later2:
#                 duplicate2.append(FCGR)
#             else:
#                 keep_for_later2.append(FCGR.replace(substrings2[0], "!!!"))
#         elif substrings2[1] in FCGR:
#             if FCGR.replace(substrings2[1], "!!!") in keep_for_later2:
#                 duplicate2.append(FCGR)
#             else:
#                 keep_for_later2.append(FCGR.replace(substrings2[1], "!!!"))
#     isolate = root.split("\\")[-2]
#     print("\n###")
#     print(isolate)
#     print("###")
#     # Delete the duplicate folder. Keep the one with the substring _1.0_3_. If there is no such folder, keep the one with the substring _1_3_ and rename it to _1.0_3_
#     for FCGR in FCGRs:
#         if FCGR in duplicate1:
#             if substrings1[1] in FCGR: # make sure to delete the folder with _1_3_ and not the one with _1.0_3_
#                 FCGR.replace(substrings1[1], substrings1[0])
#             # else: # if the folder has the substring _1_3_ then delete it
#             print(f"deleting {FCGR} in {isolate}")
#             shutil.rmtree(os.path.join(root, FCGR), ignore_errors=True)
#
#         elif FCGR in duplicate2:
#             # make sure to delete the folder with _1_1_ and not the one with _1.0_1_
#             if substrings2[1] in FCGR:
#                 FCGR.replace(substrings2[1], substrings2[0])
#             # else: # if the folder has the substring _1_1_ then delete it
#             print(f"deleting {FCGR} in {isolate}")
#             shutil.rmtree(os.path.join(root, FCGR), ignore_errors=True)
#         elif substrings1[0] in FCGR:
#             # rename the folder to _1.0_3_
#             new_name = FCGR.replace(substrings1[0], substrings1[1])
#             print(f"renaming {FCGR} to {new_name} in {isolate}")
#             os.rename(os.path.join(root, FCGR), os.path.join(root, new_name))
#         elif substrings2[0] in FCGR:
#             # rename the folder to _1.0_1_
#             new_name = FCGR.replace(substrings2[0], substrings2[1])
#             print(f"renaming {FCGR} to {new_name} in {isolate}")
#             os.rename(os.path.join(root, FCGR), os.path.join(root, new_name))


            # new_name = FCGR.replace(substrings1[0], substrings1[1])
            # os.rename(os.path.join(root, FCGR), os.path.join(root, new_name))
            # CHECK if the other substring exists in the same folder


    # check if two dirs are the same except for the substring (either _1_3_ and _1.0_3_ or _1_1_ and _1.0_1_)
    # sub1:


    # indexes_with_103 = [i for i, s in enumerate(dirs) if substrings1[1] in s]
    # set1 = set(indexes_with_13).intersection(indexes_with_103)
    # # print(f"{root}, {dirs}, {files[list(set1)]}")
    # print(set1)
    # # sub2:
    # indexes_with_11 = [i for i, s in enumerate(dirs) if substrings2[0] in s]
    # indexes_with_101 = [i for i, s in enumerate(dirs) if substrings2[1] in s]
    # set2 = set(indexes_with_11).intersection(indexes_with_101)
    #

#%%
