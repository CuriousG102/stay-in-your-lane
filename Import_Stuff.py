import pip

def Import(package):
    pip.main(['install',package])

if __name__ == '__main__':

    package = 'matplotlib'
    Import(package)
