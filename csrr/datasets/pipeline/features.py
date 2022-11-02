import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Cumulants:
    def __init__(self):
        pass

    def __call__(self, results):
        iq = results['iqs']
        iq = iq.reshape(2, -1)
        iq = iq[0, :] + 1j * iq[1, :]
        iq_length = iq.shape[0]
        M20 = np.sum(np.power(iq, 2)) / iq_length
        M21 = np.sum(np.power(np.abs(iq), 2)) / iq_length
        M22 = np.sum(np.power(np.conj(iq), 2)) / iq_length
        M40 = np.sum(np.power(iq, 4)) / iq_length
        M41 = np.sum(np.power(iq, 2) * np.power(np.abs(iq), 2)) / iq_length
        M42 = np.sum(np.power(np.abs(iq), 4)) / iq_length
        M43 = np.sum(np.power(np.conj(iq), 2) * np.power(np.abs(iq), 2)) / iq_length
        M60 = np.sum(np.power(iq, 6)) / iq_length
        M61 = np.sum(np.power(iq, 4) * np.power(np.abs(iq), 2)) / iq_length
        M62 = np.sum(np.power(iq, 2) * np.power(np.abs(iq), 4)) / iq_length
        M63 = np.sum(np.power(np.abs(iq), 6)) / iq_length

        C20 = M20
        C21 = M21
        C40 = M40 - 3 * np.power(M20, 2)
        C41 = M41 - 3 * M20 * M21
        C42 = M42 - np.power(np.abs(M20), 2) - 2 * np.power(M21, 2)
        C60 = M60 - 15 * M20 * M40 + 3 * np.power(M20, 3)
        C61 = M61 - 5 * M21 * M40 - 10 * M20 * M41 + 30 * np.power(M20, 2) * M21
        C62 = M62 - 6 * M20 * M42 - 8 * M21 * M41 - M22 * M40 + 6 * np.power(M20, 2) * M22 + 24 * np.power(M21, 2) * M20
        C63 = M63 - 9 * M21 * M42 + 12 * np.power(M21, 3) - 3 * M20 * M43 - 3 * M22 * M41 + 18 * M20 * M21 * M22

        C20_norm = C20 / np.power(C21, 2)
        C21_norm = C21 / np.power(C21, 2)
        C40_norm = C40 / np.power(C21, 2)
        C41_norm = C41 / np.power(C21, 2)
        C42_norm = C42 / np.power(C21, 2)
        C60_norm = C60 / np.power(C21, 2)
        C61_norm = C61 / np.power(C21, 2)
        C62_norm = C62 / np.power(C21, 2)
        C63_norm = C63 / np.power(C21, 2)

        cumulants = np.zeros((1, 9), dtype=np.float64)
        cumulants[0, 0] = np.abs(C20_norm)
        cumulants[0, 1] = np.abs(C21_norm)
        cumulants[0, 2] = np.abs(C40_norm)
        cumulants[0, 3] = np.abs(C41_norm)
        cumulants[0, 4] = np.abs(C42_norm)
        cumulants[0, 5] = np.abs(C60_norm)
        cumulants[0, 6] = np.abs(C61_norm)
        cumulants[0, 7] = np.abs(C62_norm)
        cumulants[0, 8] = np.abs(C63_norm)

        results['cls'] = cumulants
        return results
