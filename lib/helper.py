import os
def makedir(dirname):
    path = os.path.join(os.getcwd(), dirname)
    if not os.path.exists(path):
        os.makedirs(path)