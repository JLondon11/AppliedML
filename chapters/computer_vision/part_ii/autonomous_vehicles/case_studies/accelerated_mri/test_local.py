"""Synthetic fixture unit tests only. Not fastMRI clinical or model evidence."""
import tempfile
from pathlib import Path
import h5py,numpy as np
from evaluate_fastmri import centered_fft2,centered_ifft2,cartesian_mask,pocs_tv,run_file,assess
def main():
    rng=np.random.default_rng(551)
    im=rng.normal(size=(2,64,80))+1j*rng.normal(size=(2,64,80))
    assert np.allclose(centered_ifft2(centered_fft2(im)),im,atol=1e-10)
    mask=cartesian_mask((64,80),4,.08)
    measured=centered_fft2(im)*mask[None]
    estimate=pocs_tv(measured,mask,iterations=2)
    assert np.allclose(centered_fft2(estimate)[:,mask],measured[:,mask],atol=1e-7)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'fixture.h5'
        with h5py.File(p,'w') as f:f.create_dataset('kspace',data=np.stack([centered_fft2(im)]))
        r=run_file(p,0)
        assert r['reference'].shape==(64,80)
        assert np.isfinite(assess(r['zero_fill'],r['reference'])).all()
    print('PASS Fourier inversion, acquired data consistency, raw-file contract')
if __name__=='__main__':main()
