import sys
import logging
import unittest
from unittest import mock


import utils

class UtilsTest(unittest.TestCase):

    def setUp(self):
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.INFO)
        self.ch.setFormatter(fmt)
        # self.logger = logging.getLogger(__name__)
        self.logger = logging.getLogger()  # the root logger
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.ch)

    def tearDown(self):
        self.logger.removeHandler(self.ch)

    def test_logger(self):
        logger = utils.Logger().get()
        def run_log():
            logger.info('hello world!')
        with mock.patch.object(logger, 'info') as logger_info:
            run_log()
            logger_info.assert_called_once_with('hello world!')

    def test_running_avg(self):
        running_avg = utils.RunningAvg()
        for i in range(10):
            running_avg.step(i)
        self.assertEqual(running_avg(), sum(range(10))/10)

        running_avg.reset()
        self.assertIsNone(running_avg.data)
        self.assertEqual(running_avg.steps, 0)

        running_avg.step((1, 1, 1))
        running_avg.step((2, 2, 2))
        self.assertSequenceEqual(running_avg(), (1.5, 1.5, 1.5))
        self.assertListEqual(running_avg(), [1.5, 1.5, 1.5])

        running_avg.reset()
        running_avg.step({'loss': 0.2, 'accuracy': 0.8})
        running_avg.step({'loss': 0.3, 'accuracy': 0.9})
        running_avg.step({'loss': 0.4, 'accuracy': 1.0})
        self.assertDictEqual(running_avg(), {'loss': 0.3, 'accuracy': 0.9})

        # don't reset and add float into running_avg
        with self.assertRaises(Exception):
            running_avg.step(1.2)

        # add str type will raise exception
        running_avg.reset()
        with self.assertRaises(Exception):
            running_avg.step('abc')

    def test_progress_bar(self):
        for idx, bar, item in utils.ProgressBarWrapper(range(100), 100):
            if item == 98:
                bar.set_suffix('Complete!')
            self.logger.debug(item)

    def test_table(self):
        table = utils.Table()
        for i in range(10):
            table.append(list(range(20)))
        self.assertSequenceEqual(table.shape, (10, 20))

        table.clear()
        self.assertEqual(table.num_rows, 0)

        r1 = {'c1': 1, 'c2': 2}
        r2 = {'c1': 2, 'c2': 4}
        r3 = {'c1': 4, 'c2': 8}
        mu = {'c1': (1+2+4)/3.0, 'c2': (2+4+8)/3.0}
        table.append(r1)
        table.append(r2)
        table.append(r3)
        self.assertSequenceEqual(table.shape, (3, 2))
        self.assertDictEqual(table.row(0), r1)
        self.assertDictEqual(table.row(-1), r3)
        self.assertSequenceEqual(table.column('c1'), (1, 2, 4))
        self.assertSequenceEqual(table.column('c2'), (2, 4, 8))

        self.assertDictEqual(table.max('c1'), r3)
        self.assertDictEqual(table.max('c2'), r3)
        self.assertDictEqual(table.min('c1'), r1)
        self.assertDictEqual(table.mean(['c1', 'c2']), mu)
        
        def f(row):
            if row['c1'] % 2 == 0:
                return None
            row['c1'] = row['c1'] ** 2
            row['c2'] = row['c2'] ** 2
            return row
        g = list(table.filter(f))
        self.assertDictEqual(g[0], {'c1': 1, 'c2': 4})

        table.clear()
        table.append(r1)
        txt = table.csv()
        # self.assertEqual('c1,c2\r\n1,2\r\n', txt)
        self.logger.debug(txt)
        txt = table.tabulate()
        self.logger.debug(txt)