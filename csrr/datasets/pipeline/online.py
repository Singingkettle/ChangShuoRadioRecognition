import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class GeneratePSDFromIQ:
    """
    Estimate power spectral density using a periodogram from IQ data
    """

    def __init__(self,
                 to_float32=False,
                 to_norm=False):
        self.to_float32 = to_float32
        self.to_norm = to_norm

    def __call__(self, results):
        iqs = results['inputs']['iqs']
        iqs = iqs[0, :] + 1j * iqs[1, :]
        N = iqs.shape[0]
        xdft = np.fft.fft(iqs)
        psdx = 1 / (2 * np.pi * N) * np.power(np.abs(xdft), 2)
        psd = 10 * np.log10(psdx)

        results['inputs']['psds'] = psd

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'to_norm={self.to_norm}, )')
        return repr_str
