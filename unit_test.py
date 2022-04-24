import unittest
import jpx-transformer

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()



def test_prep_dateset():
    (train_ds, val_ds, time_series) = prep_dataset( prices, 20, 32 )
   # print(len( val_ds ))
    for (x, y) in val_ds:
        print("x=", x.numpy().shape, "y=", y.numpy().shape)
    print()

#test_prep_dateset()

