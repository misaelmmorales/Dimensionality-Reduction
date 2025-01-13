import numpy as np
from pywt import wavedec2, waverec2, coeffs_to_array, array_to_coeffs
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

df_perm = np.load('data/data_500_128x128.npy')

keep_percs = np.linspace(0.01, 0.99, 100)
reconstructions = {'xhat':[], 'coeffs':[], 'psnr': [], 'mse': [], 'ssim': []}

wavelet = 'haar'
levels = 1

for i, k in enumerate(keep_percs):
    coeffs = wavedec2(df_perm, wavelet=wavelet, level=levels, axes=(-2, -1))
    coeff_arr, coeff_slices = coeffs_to_array(coeffs, axes=(-2,-1))
    coeff_sort = np.sort(np.abs(coeff_arr.reshape(-1)))

    threshold = coeff_sort[int(np.floor((1-k)*len(coeff_sort)))]
    coeff_arr_t = coeff_arr * (np.abs(coeff_arr) > threshold)
    coeffs_t = array_to_coeffs(coeff_arr_t, coeff_slices, output_format='wavedec2')
    
    reconstructions['xhat'].append(waverec2(coeffs_t, wavelet=wavelet))
    reconstructions['coeffs'].append(coeff_arr)

    psnr = peak_signal_noise_ratio(df_perm, reconstructions['xhat'][-1], data_range=1)
    mse = mean_squared_error(df_perm, reconstructions['xhat'][-1])
    ssim = structural_similarity(df_perm, reconstructions['xhat'][-1], data_range=1)

    reconstructions['psnr'].append(psnr)
    reconstructions['mse'].append(mse)
    reconstructions['ssim'].append(ssim)