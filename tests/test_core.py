"""Small hand-computable cases cover semantics, ties and negative pools."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from loveda_rescue.data import map_labels
from loveda_rescue.rescue import rescue, effects, score
from loveda_rescue.evaluation import confusion_matrix, summarize
from loveda_rescue.calibration import calibrate
from loveda_rescue.spatial import spatial_prediction

class CoreTests(unittest.TestCase):
    def test_encoding(self):
        np.testing.assert_array_equal(map_labels(np.arange(8)),[255,0,1,2,3,4,5,6])
        with self.assertRaises(ValueError): map_labels(np.array([8]))
    def test_candidates_ties_and_threshold(self):
        p=np.array([.5,.2,.2,.025,.025,.025,.025],np.float32)[:,None,None]
        self.assertEqual(int(rescue(p,1,.2,'probability')[0,0]),1)
        self.assertEqual(int(rescue(p,2,.2,'probability')[0,0]),0)
        self.assertEqual(int(rescue(p,1,.20001,'probability')[0,0]),0)
        p[0]=.1; p[6]=.425
        self.assertEqual(int(rescue(p,1,0,'probability')[0,0]),6)
    def test_bhw_and_ignore(self):
        gt=np.array([[1,0,4,255]]); b=np.zeros_like(gt); a=np.ones_like(gt)
        self.assertEqual(effects(gt,b,a),dict(B=1,H=1,W=1,M=3,net=0))
        self.assertEqual(effects(gt,b,b)['M'],0)
    def test_metrics(self):
        cm=confusion_matrix(np.array([0,1,1,255]),np.array([0,1,0,6]))
        m=summarize(cm)
        self.assertAlmostEqual(m['OA'],2/3); self.assertEqual(m['IoU'][:2],[.5,.5])
        self.assertIsNone(m['Precision'][3]); self.assertEqual(m['mIoU'],.5)
    def test_all_valid_negative_pool(self):
        # Pixel 2 is not a background candidate but must enter the negative pool.
        p=np.full((7,1,3),.01,np.float32); p[0]=[.5,.1,.5]; p[1]=[.2,.8,.3]
        p/=p.sum(0,keepdims=True); gt=np.array([[0,2,255]])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.npz'; np.savez(path,prob=p,gt=gt)
            tau=calibrate([path],1,'probability')
        self.assertAlmostEqual(tau,float(np.quantile(p[1,0,:2],.999)),places=7)
    def test_c4_cannot_create_missing_class(self):
        p=np.full((7,8,8),.01,np.float32); p[0]=.5; p[1]=.45; p/=p.sum(0,keepdims=True)
        result=spatial_prediction(p,np.zeros((8,8,3),np.uint8),'C4',dict(support=.6,probability_gap=.05,global_threshold_bin=0,class_threshold_bins=[0]*7))
        self.assertTrue(np.all(result==0))

if __name__=='__main__': unittest.main()
