# Borrowed MTP host weights

`LLAMA_MTP_PIN_BORROWED_HOST=1` pins only successfully borrowed CPU output/embedding tensors in their donor's existing storage. GPU tensors and the PLE lookup table are excluded. Locks belong to the donor, survive repeated helper loads and stay alive through borrowed-storage ownership. A failed lock refuses helper loading; it never silently proceeds with evictable borrowed weights. The option is off until enabled by a launch configuration. It establishes residency for those tensors, not attribution of historical unknown bytes.
