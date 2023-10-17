import os


def scanRecurse(baseDir: str):
    baseDir = baseDir.split('*')[0] if  '*' in baseDir else baseDir

    for entry in os.scandir(baseDir):
        if entry.is_file():
            yield os.path.join(baseDir, entry.name)
        else:   # recurse needs from, otherwise generator object is returned
            yield from scanRecurse(entry.path + '/')