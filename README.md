# OFI

This project constructs the various versions of Order Flow Imbalance (OFI) features defined in the article below.

Rama Cont, Mihai Cucuringu & Chao Zhang (2023) Cross-impact of
order flow imbalance in equity markets, Quantitative Finance, 23:10, 1373-1393, DOI:
10.1080/14697688.2023.2236159

To apply the construction in the code, use/modify the following sample command in terminal:

python ofi_features.py first_25000_rows.csv AAPL 2024-10-21T11:55:00Z --window 1s
