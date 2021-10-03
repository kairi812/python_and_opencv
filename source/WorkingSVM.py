datadir = "data"
dataset = "pedestrians128x64"
datafile = "%s/%s.tar.gz" % (datadir, dataset)

extractdir = "%s/%s.tar.gz" % (datadir, dataset)

def extract_tar(datafile, extractdir):
    try:
        import tarfile
    except ImportError:
        raise ImportError("You do not have tarfile installed."
                          "Try unzipping the file outside of"
                          "Python.")
    tar = tarfile.open(datafile)
    tar.extractall(path=extractdir)
    tar.close()
    print("%s sucessfully extracted to %s" % (datafile, extractdir))

extract_tar(datafile, datadir)

# 動画がないため未確認