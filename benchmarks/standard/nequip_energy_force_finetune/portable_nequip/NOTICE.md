# Portable NequIP source notice

The files under `portable_nequip/nn`, together with the local atomic-data and model
builder modules, are adapted from the NequIP implementation vendored in the
OneScience repository. They are included here so the scored implementation is
independent and does not import OneScience model code at runtime. The evaluation
contract fixes the OneScience example checkpoint and data hashes separately.
