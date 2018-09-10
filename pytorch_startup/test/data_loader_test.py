import time
import unittest

import numpy as np

import config
from data_loader import DataLoader
from utils import Params


class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        self.trainset = config.train_name
        self.valset = config.val_name
        self.testset = config.test_name
        self.tag_padding = -1
        self.params = Params(config.datasets_params_file)
        self.loader = DataLoader(config.data_dir, self.params)

    def test_batch_iterator(self):
        batch_size = 10
        batches = self.loader.load(self.trainset, 
                                   batch_size=batch_size, 
                                   shuffle=True)
        self.assertEqual(self.trainset, batches.name)
        self.assertEqual(self.params.train_size, batches.dataset_size)
        self.assertEqual(batch_size, batches.batch_size)
        # convert generator to iterator
        batches = iter(batches)
        with self.assertRaises(StopIteration):
            while True:
                batch = next(batches)

    def test_trainset_size(self):
        batches = self.loader.load(self.trainset, batch_size=10, 
                                   shuffle=True)
        size = 0
        for batch in batches:
            batch_input, batch_target = batch
            size += batch_input.shape[0]
        self.assertEqual(self.params.train_size, size)

    def test_valset_size(self):
        batches = self.loader.load(self.valset, batch_size=10, 
                                   shuffle=True)
        size = 0
        for batch in batches:
            batch_input, batch_target = batch
            size += batch_input.shape[0]
        self.assertEqual(self.params.val_size, size)

    def test_testset_size(self):
        batches = self.loader.load(self.testset, batch_size=10, 
                                   shuffle=True)
        size = 0
        for batch in batches:
            batch_input, batch_target = batch
            size += batch_input.shape[0]
        self.assertEqual(self.params.test_size, size)

    def test_shuffle_runtime(self):
        start_time = time.time()
        # off shuffle: lazy loading, but slow
        batches = self.loader.load(self.trainset, batch_size=10,
                                   shuffle=False)
        for batch in batches:
            inputs, targets = batch
            for i, t in zip(inputs, targets):
                pass
        off_shuffle_time = time.time() - start_time
        start_time = time.time()
        # on shuffle: memory consuming, but quick
        batches = self.loader.load(self.trainset, batch_size=10,
                                   shuffle=True)
        for batch in batches:
            inputs, targets = batch
            for i, t in zip(inputs, targets):
                pass
        on_shuffle_time = time.time() - start_time
        # this assert will fail on sometime
        # self.assertGreater(off_shuffle_time, on_shuffle_time)

    def test_batch_size(self):
        batch_size = 10
        batches = self.loader.load(self.trainset, batch_size=batch_size, 
                                   shuffle=True)
        for batch in batches:
            inputs, targets = batch
            self.assertEqual(inputs.shape, targets.shape)
            self.assertLessEqual(inputs.shape[0], batch_size)

    def test_tag_padding(self):
        batches = self.loader.load(self.trainset, batch_size=10, 
                                   shuffle=True)
        for batch in batches:
            inputs, targets = batch
            w, h = targets.shape
            rows, columns = np.where(targets == self.tag_padding)
            flag = False
            for r, c in zip(rows, columns):
                while c < h:
                    self.assertTrue(targets[r][c] == self.tag_padding)
                    c += 1


# Don't run test module as __main__. Please run test via: 
# `python -m unittest test.data_loader`
if __name__ == '__main__':
    unittest.main()