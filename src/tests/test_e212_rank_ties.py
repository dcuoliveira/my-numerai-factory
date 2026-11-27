"""Rank-average ensemble ties and residual projections reproduce across batches."""
from fractions import Fraction
import numpy as np
from scipy.stats import rankdata
from experiments import e212_volatility99_tsfm as S
from src.prediction import meta_resid as MR


def test_rank_averages_match_exact_fractions_for_all_ensemble_sizes_and_layouts():
    rng=np.random.default_rng(212)
    values=np.round(rng.normal(size=(103,7)),1)
    rank_numerators=(2*rankdata(values,axis=0)-1).astype(np.int64)
    combos=S.definitions(7);members=np.zeros((7,len(combos)),np.int64)
    sizes=np.array([len(m) for m in combos])
    for j,m in enumerate(combos):members[list(m),j]=1
    full=S.rank_average_blends(rank_numerators,members,sizes)
    expected=np.array([[float(Fraction(sum(int(rank_numerators[r,c]) for c in m),2*len(values)*len(m))) for m in combos] for r in range(len(values))])
    np.testing.assert_array_equal(full,expected)
    np.testing.assert_array_equal(full,S.rank_average_blends(np.asfortranarray(rank_numerators),np.asfortranarray(members),sizes))
    for start in range(0,len(combos),17):
        np.testing.assert_array_equal(full[:,start:start+17],S.rank_average_blends(rank_numerators,members[:,start:start+17],sizes[start:start+17]))
    # Several distinct rank permutations deliberately have the same average.
    tied=np.array([[1,3,5],[5,3,1],[3,5,1]],np.int64)
    blend=S.rank_average_blends(tied,np.ones((3,1),np.int64),[3])
    np.testing.assert_array_equal(blend,np.full((3,1),.5))


def test_residual_projection_is_exactly_columnwise_and_native_equivalent():
    rng=np.random.default_rng(14);pred=np.round(rng.normal(size=(103,31)),1);meta=rng.normal(size=103);meta[:9]=np.nan
    batch,beta=S.residual_matrix(pred,meta)
    for j in range(pred.shape[1]):
        one,b=S.residual_matrix(pred[:,j,None],meta)
        np.testing.assert_array_equal(batch[:,j],one[:,0]);np.testing.assert_array_equal(beta[j],b[0])
        native,native_beta=MR.residualise(pred[:,j],meta)
        np.testing.assert_allclose(batch[:,j],native,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(beta[j],native_beta,rtol=1e-12,atol=1e-12)
