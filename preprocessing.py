
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import collections

def split_files(base_dir):
    print('Spliting data between test and train dirs')

    # Only split files if haven't already
    if not os.path.isdir(base_dir + 'test') and not os.path.isdir(base_dir + 'train'):

        def copytree(src, dst, symlinks = False, ignore = None):
            if not os.path.exists(dst):
                os.makedirs(dst)
                shutil.copystat(src, dst)
            lst = os.listdir(src)
            if ignore:
                excl = ignore(src, lst)
                lst = [x for x in lst if x not in excl]
            for item in lst:
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if symlinks and os.path.islink(s):
                    if os.path.lexists(d):
                        os.remove(d)
                    os.symlink(os.readlink(s), d)
                    try:
                        st = os.lstat(s)
                        mode = stat.S_IMODE(st.st_mode)
                        os.lchmod(d, mode)
                    except:
                        pass # lchmod not available
                elif os.path.isdir(s):
                    copytree(s, d, symlinks, ignore)
                else:
                    shutil.copy2(s, d)

        def generate_dir_file_map(path):
            dir_files = collections.defaultdict(list)
            with open(path, 'r') as txt:
                files = [l.strip() for l in txt.readlines()]
                for f in files:
                    dir_name, id = f.split('/')
                    dir_files[dir_name].append(id + '.jpg')
            return dir_files

        train_dir_files = generate_dir_file_map(base_dir + 'meta/train.txt')
        test_dir_files = generate_dir_file_map(base_dir + 'meta/test.txt')


        def ignore_train(d, filenames):
            print(d)
            subdir = d.split('/')[-1]
            to_ignore = train_dir_files[subdir]
            return to_ignore

        def ignore_test(d, filenames):
            print(d)
            subdir = d.split('/')[-1]
            to_ignore = test_dir_files[subdir]
            return to_ignore

        copytree(base_dir + 'images', base_dir + 'test', ignore=ignore_train)
        copytree(base_dir + 'images', base_dir + 'train', ignore=ignore_test)

    else:
        print('Train/Test files already copied into separate folders.')

def map_classes(base_dir):
    class_to_ix = {}
    ix_to_class = {}
    with open(base_dir + 'meta/classes.txt', 'r') as txt:
        classes = [l.strip() for l in txt.readlines()]
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
    sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

    return ix_to_class


