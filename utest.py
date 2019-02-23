import unittest
import preproc
import numpy as np

class u_test(unittest.TestCase):

    def test_mean(self):
        test_img = np.array([[1,2,3],[1,2,3],[1,2,3]])
        result = preproc.mean(test_img)
        expected_result = 2.0
        self.assertEqual(result, expected_result )
 
if __name__ == '__main__':
    unittest.main()
