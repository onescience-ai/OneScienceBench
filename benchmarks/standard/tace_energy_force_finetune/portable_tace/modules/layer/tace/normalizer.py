# from typing import Union


# import torch


# class Normalizer(torch.nn.Module):

#     def __init__(
#         self,
#         mean: Union[torch.nn.Module, float] = 0.0,
#         rmsd: Union[torch.nn.Module, float] = 1.0,
#     ):
#         super().__init__()

#         if isinstance(mean, float):
#             mean = torch.tensor(mean)
#         if isinstance(rmsd, float):
#             rmsd = torch.tensor(rmsd)

#         self.register_buffer(name="mean", tensor=mean)
#         self.register_buffer(name="rmsd", tensor=rmsd)

#     def norm(self, tensor: torch.Tensor) -> torch.Tensor:
#         return (tensor - self.mean) / self.rmsd

#     def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
#         return normed_tensor * self.rmsd + self.mean

#     def forward(self, normed_tensor: torch.Tensor) -> torch.Tensor:
#         return self.denorm(normed_tensor)

#     def __repr__(self):
#         mean = self.mean.item()
#         rmsd = self.rmsd.item()
#         return f"{self.__class__.__name__}(mean={mean:.4f}, rmsd={rmsd:.4f})"
