import os

path = "dataset/"


def get_alphabet_value(code):
    if code <= 10:
        value = code - 1
        return str(value)
    else:
        if code <= 36:
            value = code - 11 + 65
            return chr(value)
        else:
            value = code - 37 + 97
            return chr(value)


for dir_ in os.listdir(path):
    code = dir_[-3 : len(dir_)]
    os.rename(path + dir_, path + get_alphabet_value(int(code)))


for dir_ in os.listdir(path):
    count = 0
    for image in os.listdir(path + dir_):
        count += 1
        os.rename(
            path + dir_ + "/" + image, path + dir_ + "/" + dir_ + "_" + str(count)
        )
