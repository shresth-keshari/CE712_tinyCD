"""Numbers *reported* in the TinyCD paper, transcribed verbatim (all metrics in %).

Codegoni, Lombardi & Ferrari, "TinyCD: A (Not So) Deep Learning Model for Change Detection",
Neural Computing and Applications 35:8471-8486 (2023), DOI 10.1007/s00521-022-08122-3.

Nothing in this module is measured by us. Every value carries the table it was copied from, so the app can
always label it "Reported (Table N)".
"""

CITATION = (
    "A. Codegoni, G. Lombardi, A. Ferrari, \"TinyCD: A (Not So) Deep Learning Model for Change Detection\", "
    "Neural Computing and Applications 35, 8471-8486 (2023). DOI 10.1007/s00521-022-08122-3"
)
DOI_URL = "https://doi.org/10.1007/s00521-022-08122-3"
CODE_URL = "https://github.com/AndreaCodegoni/Tiny_model_4_CD"

METRICS = ("precision", "recall", "f1", "iou", "oa")

# Table 1 - LEVIR-CD test set.
TABLE1_LEVIR = {
    "FC-EF":         dict(precision=86.91, recall=80.17, f1=83.40, iou=71.53, oa=98.39),
    "FC-Siam-diff":  dict(precision=89.53, recall=83.31, f1=86.31, iou=75.92, oa=98.67),
    "FC-Siam-conc":  dict(precision=91.99, recall=76.77, f1=83.69, iou=71.96, oa=98.49),
    "DTCDSCN":       dict(precision=88.53, recall=86.83, f1=87.67, iou=78.05, oa=98.77),
    "STANet":        dict(precision=83.81, recall=91.00, f1=87.26, iou=77.40, oa=98.66),
    "IFNet":         dict(precision=94.02, recall=82.93, f1=88.13, iou=78.77, oa=98.87),
    "SNUNet":        dict(precision=89.18, recall=87.17, f1=88.16, iou=78.83, oa=98.82),
    "BIT":           dict(precision=89.24, recall=89.37, f1=89.31, iou=80.68, oa=98.92),
    "ChangeFormer":  dict(precision=92.05, recall=88.80, f1=90.40, iou=82.48, oa=99.04),
    "TinyCD":        dict(precision=92.68, recall=89.47, f1=91.05, iou=83.57, oa=99.10),
}

# Table 2 - WHU-CD test set (not evaluated in this project; shown for context only).
TABLE2_WHU = {
    "BIT":    dict(precision=86.64, recall=81.48, f1=83.98, iou=72.39, oa=98.75),
    "TinyCD": dict(precision=91.72, recall=91.76, f1=91.74, iou=84.74, oa=99.34),
}

# Table 3 - parameters (millions), parameter ratio vs TinyCD, GFLOPs. ChangeFormer FLOPs are "N.D" in the paper.
TABLE3_SIZE = {
    "DTCDSCN":      dict(params_m=41.07, ratio=146.67, gflops=7.21),
    "STANet":       dict(params_m=16.93, ratio=60.46, gflops=6.58),
    "IFNet":        dict(params_m=50.71, ratio=181.10, gflops=41.18),
    "SNUNet":       dict(params_m=12.03, ratio=42.96, gflops=27.44),
    "BIT":          dict(params_m=3.55, ratio=12.67, gflops=4.35),
    "ChangeFormer": dict(params_m=41.02, ratio=146.50, gflops=None),
    "TinyCD":       dict(params_m=0.28, ratio=1.0, gflops=1.45),
}

# Table 4 - backbone slicing and final PW-MLP classifier (LEVIR-CD).
TABLE4_LEVIR = {
    "Full w/o MLP":   dict(precision=83.05, recall=94.00, f1=88.19, iou=78.88, oa=98.71, params=17_740_598),
    "Full w MLP":     dict(precision=92.65, recall=89.26, f1=90.92, iou=83.36, oa=99.09, params=17_743_288),
    "Sliced w/o MLP": dict(precision=46.15, recall=94.52, f1=62.02, iou=44.95, oa=94.10, params=282_438),
    "Sliced w MLP":   dict(precision=92.68, recall=89.47, f1=91.05, iou=83.57, oa=99.10, params=285_128),
}

# Table 5 - with/without MAMB skip connections (LEVIR-CD). The paper gives no parameter count here.
TABLE5_LEVIR = {
    "No skip": dict(precision=92.35, recall=88.50, f1=90.38, iou=82.45, oa=99.04),
    "Skip":    dict(precision=92.68, recall=89.47, f1=91.05, iou=83.57, oa=99.10),
}

# Table 6 - fusion strategy (LEVIR-CD).
TABLE6_LEVIR = {
    "Subtraction":   dict(precision=92.13, recall=89.41, f1=90.75, iou=83.07, oa=99.07, params=282_939, gflops=1.43),
    "Concat + conv": dict(precision=92.55, recall=89.61, f1=91.06, iou=83.59, oa=99.10, params=368_468, gflops=1.75),
    "Mix (ours)":    dict(precision=92.68, recall=89.47, f1=91.05, iou=83.57, oa=99.10, params=285_128, gflops=1.45),
}

# Table 7 - fusion alternative in the skip MAMBs and/or the bottleneck (F1 on LEVIR-CD = "F1-L").
# The table marks each location with a check or a cross. Its caption says the check marks a replaced
# location, but the row with checks in both columns has 285,128 parameters, i.e. the unmodified TinyCD. We
# therefore identify each row by its parameter count, which our builds reproduce exactly (scripts/variants.py).
TABLE7_LEVIR = {
    "sub_all":        dict(f1=90.75, params=282_939),
    "sub_skips":      dict(f1=90.75, params=284_004),
    "sub_bottleneck": dict(f1=90.71, params=284_063),
    "cat_all":        dict(f1=91.06, params=368_468),
    "cat_skips":      dict(f1=91.06, params=313_028),
    "cat_bottleneck": dict(f1=90.90, params=340_568),
    "paper":          dict(f1=91.05, params=285_128),
}

# Table 9 - backbone comparison (LEVIR-CD), F1 and parameter count.
TABLE9_LEVIR = {
    "mobilenetv2":      dict(f1=88.63, params=38_798),
    "mobilenetv3large": dict(f1=88.21, params=32_886),
    "resnet18":         dict(f1=89.72, params=707_894),
    "efficientnetb0":   dict(f1=90.02, params=79_480),
    "efficientnetb1":   dict(f1=90.51, params=122_092),
    "efficientnetb2":   dict(f1=90.68, params=148_040),
    "efficientnetb3":   dict(f1=90.95, params=178_716),
    "efficientnetb4":   dict(f1=91.05, params=285_128),
    "mnasnet13":        dict(f1=90.02, params=97_262),
    "densenet121":      dict(f1=90.00, params=1_364_790),
}
