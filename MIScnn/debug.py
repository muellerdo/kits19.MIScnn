from utils.nifti_io import load_segmentation_nii
import statistics

cases = list(range(0,210))
data_path = "/home/mudomini/projects/KITS_challenge2019/kits19/data.original"

res = []
max = 0
min = 10000000
for i in cases:
    seg = load_segmentation_nii(i, data_path)
    seg = seg.get_data()
    x = seg.shape[0]
    y = seg.shape[1]
    z = seg.shape[2]
    res.append(x)
    if x > max:
        max = x
    if x < min:
        min = x
    print(i, x, y, z)

print("##########################################")
ds_mean = statistics.mean(res)
ds_median = statistics.median(res)

print("Mean: " + str(ds_mean))
print("Median: " + str(ds_median))
print("Max:" + str(max))
print("Min:" + str(min))
