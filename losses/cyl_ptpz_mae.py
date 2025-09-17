import torch

#based on Diptarko's code
#measures error in pT and pZ
class CylPtPzMAE(torch.nn.Module):
    def __init__(self, norm_scales, norm_biases):
        super().__init__()

        #convert numpy arrays (scales and biases) to torch tensors
        s = torch.as_tensor(norm_scales, dtype=torch.float32)
        b = torch.as_tensor(norm_biases, dtype=torch.float32)

        #reshape scales and biases to 3 features (pt, eta, phi or E) if 1D
        if s.ndim == 1:
            assert s.numel() % 3 == 0, "Scale length must be divisible by 3"
            n = s.numel() // 3
            s = s.view(n, 3)
            b = b.view(n, 3)
        else:
            #if not 1D, still shape to 3 features
            n = s.shape[0]
            assert s.shape == b.shape == (n, 3)

        #num of particle constituents per event
        self.n_const = n

        #store scales and biases as buffers (not parameters, but saved in the model state)
        self.register_buffer("scales", s)
        self.register_buffer("biases", b)

    def forward(self, y_pred_flat: torch.Tensor, y_true_flat: torch.Tensor) -> torch.Tensor:

        B, F = y_true_flat.shape

        #checks that the total features = 3 * number of constituents
        assert F == 3 * self.n_const, f"expected {3*self.n_const}, got {F}"

        #reshapes (B, 3*N) to (B, N, 3)
        y_p = y_pred_flat.view(B, self.n_const, 3).float()
        y_t = y_true_flat.view(B, self.n_const, 3).float()

        #move scales and biases to device
        s = self.scales.to(y_t.device)
        b = self.biases.to(y_t.device)

        #denormalized = normalized * scale + bias
        y_t_den = y_t * s + b
        y_p_den = y_p * s + b

        #get pt (index 0) and eta (index 1) for truth and predictions
        pt_t = y_t_den[:, :, 0]  # (B, N)
        eta_t = y_t_den[:, :, 1]
        pt_p = y_p_den[:, :, 0]
        eta_p = y_p_den[:, :, 1]

        # Safety clamps:
        #clamp eta to avoid huge sinh() blowups
        eta_t = torch.clamp(eta_t, -8.0, 8.0)
        eta_p = torch.clamp(eta_p, -8.0, 8.0)

        #pz = pt*sinh(eta)
        pz_t = pt_t * torch.sinh(eta_t)
        pz_p = pt_p * torch.sinh(eta_p)

        #calculate error
        err = (pt_t - pt_p).abs() + (pz_t - pz_p).abs()

        #return mean error
        return err.mean(dim=1)
