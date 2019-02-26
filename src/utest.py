import unittest
import preproc
import numpy as np

class u_test(unittest.TestCase):

    def test_mean_center(self):
        test_img = np.array([[1,2,3],[1,2,3],[1,2,3]])
        result = preproc.mean_center(test_img)
        expected_result = np.array([[1,1,-1],[1,0,-1],[1,-1,-1]])
        self.assertEqual(result.all(), expected_result.all())


 
if __name__ == '__main__':
    unittest.main()
