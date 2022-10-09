data_bin_num = 10
paths = []
for i in range(data_bin_num):
    paths.append("/path/to/NER/flores/20M/data-bin-split10/data-bin{}/".format(i))

paths = ":".join(paths)
print(paths)
