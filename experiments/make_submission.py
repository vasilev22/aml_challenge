"""Convert id,x,y predictions to the competition's id,prediction CSV."""
import argparse
import json
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',default='test.csv')
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',default='submission.csv')
    a=p.parse_args();test=pd.read_csv(a.test);pred=pd.read_csv(a.predictions)
    if set(pred.columns)!={'id','x','y'} or pred.id.duplicated().any() or pred.id.isna().any():
        raise ValueError('Expected unique, nonempty id,x,y rows')
    if len(pred)!=len(test) or set(pred.id)!=set(test.id):raise ValueError('IDs must match test.csv')
    xy=pred.set_index('id').loc[test.id,['x','y']].to_numpy(dtype=float)
    if not np.isfinite(xy).all():raise ValueError('Coordinates must be finite')
    pd.DataFrame({'id':test.id,'prediction':[json.dumps(v.tolist(),allow_nan=False) for v in xy]}).to_csv(a.output,index=False)


if __name__=='__main__':main()
