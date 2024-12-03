# FedDEO
The pytorch implementation of FedDEO: Description-Enhanced One-Shot Federated Learning with Diffusion Models

FedDEO is a Description-Enhanced One-Shot Federated Learning Method with Diffusion Models, offering a novel exploration of utilizing the DM in OSFL. The core idea of FedDEO involves training local descriptions on the clients, serving as the medium to transfer the knowledge of the distributed clients to the server.

## Requirements

	pip install -r requriements.txt

## Generate Images & Fine-tune & Test

	bash ./generate_test.sh
	bash ./aggregate.sh

# BibTex

	@inproceedings{10.1145/3664647.3681490,
	author = {Yang, Mingzhao and Su, Shangchao and Li, Bin and Xue, Xiangyang},
	title = {FedDEO: Description-Enhanced One-Shot Federated Learning with Diffusion Models},
	year = {2024},
	isbn = {9798400706868},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3664647.3681490},
	doi = {10.1145/3664647.3681490},
	booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
	pages = {6666–6675},
	numpages = {10},
	keywords = {diffusion model, one-shot federated learning},
	location = {Melbourne VIC, Australia},
	series = {MM '24}
	}

[Arxiv Link](https://dl.acm.org/doi/10.1145/3664647.3681490)