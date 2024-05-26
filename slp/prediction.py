from __future__ import print_function
import sys
import os
import pickle
import json
import numpy as np

def predictionRF(self, fname, predictor):
        #if os.path.isdir(persistencedir) is False:
        #    if os.makedirs(persistencedir) is False:
        #        raise OSError('Directory ' + persistencedir + ' can not be created.')
            
        #print(persistencedir)
        
        absolute_path = os.path.dirname(__file__)
        relative_path = "dump\\model\\rf\\"+fname+".pickle"
        full_path = os.path.join(absolute_path, relative_path)
        
        with open(full_path, 'rb') as f:
            self.model = pickle.load(f)
            
        p=np.array([predictor])
        y_pred = self.model.predict(p)
        y_proba = self.model.predict_proba(p)
        
        # Probabilities of the predicted response being correct.
        probabilities = y_proba[range(len(y_proba)), 0]

        result = dict()
        result['status'] = 0
        result['info'] = []
        # First column sampleids, second the prediction and third how
        # reliable is the prediction (from 0 to 1).
        result['predictions'] = np.vstack((0, y_pred, probabilities)).T.tolist()
        
        print(json.dumps(result))
        sys.exit(result['status'])
        