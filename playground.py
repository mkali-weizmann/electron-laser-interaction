import numpy as np
A = np.random.randint(0, 10, (1, 1, 3, 4))

# %%
r = np.mod(np.arange(0, -A.shape[-2], -1), A.shape[-1])
x_idx, y_idx, z_idx, t_idx = np.ogrid[:A.shape[0], :A.shape[1], :A.shape[2], :A.shape[3]]
shifted_t_idx = t_idx - r[:, np.newaxis]
shifted_t_idx_mod = np.mod(shifted_t_idx, A.shape[-1])
A_shifted = A[x_idx, y_idx, z_idx, shifted_t_idx_mod]
A_s_full_diagonals = A_shifted[:, :, :, 0:(A.shape[-1] - A.shape[-2] + 1)]

A_s_partial_diagonals = A_shifted[:, :, :, (A.shape[-1] - A.shape[-2] + 1):]
A_spd_reverse_rows = A_s_partial_diagonals[:, :, :, ::-1]
A_spdrr_tril = np.tril(A_spd_reverse_rows, k=-1)
A_spdrr_triu = np.triu(A_spd_reverse_rows, k=0)
A_spdrrtl_reordered_rows = A_spdrr_tril[:, :, :, ::-1]
A_spdrrtu_reordered_rows = A_spdrr_triu[:, :, :, ::-1]
A_spdrrturr_reversed_columns = A_spdrrtu_reordered_rows[:, :, ::-1, :]
G_sfd_cumsum = np.cumsum(A_s_full_diagonals, axis=-2)
G_spdrrtlrr_cumsum = np.cumsum(A_spdrrtl_reordered_rows, axis=-2)
G_spdrrturrrc_cumsum = np.cumsum(A_spdrrturr_reversed_columns, axis=-2)
G_spdrrturrrcc_reordered_columns = G_spdrrturrrc_cumsum[:, :, ::-1, :]
G_shifted = G_sfd_cumsum + G_spdrrtlrr_cumsum + G_spdrrturrrcc_reordered_columns



# %%
A = np.random.randint(0, 10, (2, 2, 3, 4))
# %%


