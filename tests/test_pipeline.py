"""Unit tests for the SQL cleaning/RFM logic and the Python scoring functions.

Run from the project root:  python -m unittest discover tests   (or: pytest)
The SQL tests use a tiny hand-made CSV, so they don't need the real dataset.
"""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build import connect, query  # noqa: E402
from rfm import add_rfm_scores, eta_squared, name_segment, order_clusters, score  # noqa: E402

FIXTURE = """Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country
100,11111,MUG,2,2011-11-01 10:00:00,5.0,1.0,United Kingdom
100,11111,MUG,2,2011-11-01 10:00:00,5.0,1.0,United Kingdom
101,22222,BOWL,1,2011-12-05 09:00:00,20.0,1.0,United Kingdom
101,POST,POSTAGE,1,2011-12-05 09:00:00,18.0,1.0,United Kingdom
C102,22222,BOWL,-1,2011-12-06 09:00:00,20.0,1.0,United Kingdom
103,33333,LAMP,1,2011-06-01 12:00:00,50.0,2.0,France
C104,33333,LAMP,-1,2011-06-02 12:00:00,50.0,2.0,France
105,44444,CARD,10,2011-12-01 08:00:00,1.0,,United Kingdom
106,55555,FREE GIFT,1,2011-12-01 08:00:00,0.0,3.0,United Kingdom
107,66666,OLD ORDER,1,2010-11-01 08:00:00,99.0,4.0,Germany
108,77777,PENCIL,1,2011-10-01 08:00:00,0.1,5.0,United Kingdom
108,77778,ERASER,1,2011-10-01 08:00:00,0.2,5.0,United Kingdom
C109,77779,REFUND,-1,2011-10-02 08:00:00,0.3,5.0,United Kingdom
"""


class TestSQL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "fixture.csv"
        path.write_text(FIXTURE)
        cls.con = connect(path)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_cleaning_rules(self):
        lines = self.con.execute("SELECT * FROM lines ORDER BY invoice, stock_code").df()
        self.assertEqual(len(lines), 9)                        # 13 rows - dup - POST - no ID - zero price
        self.assertNotIn("POST", set(lines.stock_code))
        self.assertFalse(lines.customer_id.isna().any())
        self.assertEqual(lines.is_cancel.sum(), 3)
        self.assertEqual(lines.loc[lines.invoice == "100", "quantity"].tolist(), [2])

    def test_rfm_net_of_returns(self):
        rfm = query(self.con, "02_rfm.sql", "2011-12-10").set_index("customer_id")
        c1 = rfm.loc[1]
        self.assertEqual(c1.frequency, 2)                      # invoices 100 and 101; cancellation not counted
        self.assertEqual(c1.recency_days, 5)                   # last purchase 2011-12-05
        self.assertAlmostEqual(c1.monetary, 10.0)              # 10 + 20 - 20 returned
        self.assertAlmostEqual(c1.returns, 20.0)
        self.assertNotIn(2, rfm.index)                         # everything returned: net 0, excluded
        self.assertNotIn(4, rfm.index)                         # outside the 365-day window
        # 0.1 + 0.2 - 0.3 is 5.6e-17 in floating point, not 0: must still count as fully refunded
        self.assertNotIn(5, rfm.index)

    def test_next_year_window(self):
        nxt = query(self.con, "03_next_year.sql", "2010-12-10").set_index("customer_id")
        self.assertEqual(nxt.loc[1, "next_orders"], 2)
        self.assertNotIn(4, nxt.index)                         # bought before the snapshot only


class TestScoring(unittest.TestCase):
    def test_scores_in_range_and_monotone(self):
        s = pd.Series(np.arange(100))
        sc = score(s)
        self.assertEqual(sc.min(), 1)
        self.assertEqual(sc.max(), 5)
        self.assertTrue(sc.is_monotonic_increasing)
        self.assertEqual(sc.value_counts().tolist(), [20] * 5)

    def test_ties_share_a_score(self):
        s = pd.Series([1] * 40 + list(range(2, 62)))
        sc = score(s)
        self.assertEqual(sc[:40].nunique(), 1)

    def test_recency_is_reversed(self):
        sc = score(pd.Series([1, 10, 100, 200, 300]), higher_is_better=False)
        self.assertEqual(sc.tolist(), [5, 4, 3, 2, 1])

    def test_every_grid_cell_has_one_segment(self):
        for r, f in itertools.product(range(1, 6), repeat=2):
            self.assertIsInstance(name_segment(r, f), str)

    def test_known_cells(self):
        self.assertEqual(name_segment(5, 5), "Champions")
        self.assertEqual(name_segment(5, 1), "New customers")
        self.assertEqual(name_segment(1, 5), "Can't lose")
        self.assertEqual(name_segment(1, 1), "Hibernating")
        self.assertEqual(name_segment(3, 3), "Need attention")

    def test_add_rfm_scores(self):
        df = pd.DataFrame({"customer_id": range(5), "recency_days": [1, 5, 50, 200, 300],
                           "frequency": [20, 10, 3, 2, 1], "monetary": [5000, 900, 300, 100, 20]})
        out = add_rfm_scores(df)
        self.assertEqual(out.loc[0, "segment"], "Champions")
        self.assertEqual(out.loc[4, "segment"], "Hibernating")
        self.assertEqual(out.loc[0, "rfm_code"], "555")


class TestHelpers(unittest.TestCase):
    def test_order_clusters_by_spend(self):
        df = pd.DataFrame({"monetary": [10, 12, 1000, 1100, 100, 90]})
        labels = np.array([7, 7, 3, 3, 0, 0])
        self.assertEqual(order_clusters(df, labels).tolist(), [3, 3, 1, 1, 2, 2])

    def test_eta_squared_bounds(self):
        y = pd.Series([1.0, 1.0, 5.0, 5.0])
        self.assertAlmostEqual(eta_squared(pd.Series(["a", "a", "b", "b"]), y), 1.0)
        self.assertAlmostEqual(eta_squared(pd.Series(["a", "b", "a", "b"]), y), 0.0)


if __name__ == "__main__":
    unittest.main()
