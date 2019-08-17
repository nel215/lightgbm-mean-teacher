import chainer
import numpy as np
from sklearn.model_selection import train_test_split
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from mean_teacher import MeanTeacherUpdater, Model, MeanTeacherChain


def test_update_core():
    np.random.seed(0)
    n_leaf = 8
    n_tree = 4
    n_batch = 10
    x = np.random.randint(0, n_leaf, size=(n_batch, n_tree))
    x += np.tile(np.arange(0, n_tree) * n_leaf, (n_batch, 1))
    y = np.random.randint(0, 2, size=(n_batch))
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=2, random_state=0)
    train_dset = TupleDataset(train_x, train_y)
    train_iter = SerialIterator(train_dset, 2)
    test_dset = test_x
    test_iter = SerialIterator(test_dset, 2)
    model = MeanTeacherChain(Model(n_tree*n_leaf))
    opt = Adam().setup(model)

    with chainer.using_config('enable_backprop', False):
        before_loss = model.teacher.forward(train_x, train_y)

    updater = MeanTeacherUpdater(
        iterator={
            'train': train_iter,
            'test': test_iter,
        },
        optimizer=opt,
        device=-1,
    )
    updater.update()

    with chainer.using_config('enable_backprop', False):
        after_loss = model.teacher.forward(train_x, train_y)

    assert before_loss.array > after_loss.array
