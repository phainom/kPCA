#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import argparse
from mpl_toolkits.mplot3d import Axes3D


import kpca


def get_data(path = '../data/train.csv'):
	t = np.loadtxt(path, delimiter=',', skiprows=1, converters={-1: lambda s: float(s[6:])})
	return t[:, 1:-1].T, np.asarray(t[:, -1])

# ------------------------------------------------------------------------------

def random_samples(X, y, instances, seed=1234):
	np.random.seed(seed)

	idxs = np.random.choice(X.shape[1], instances)

	return X[:, idxs], y[idxs]

# ------------------------------------------------------------------------------

def plot_projected(X_new, y, sub=None):
	if sub is not None:
		sub.scatter(X_new[0, :], X_new[1, :], c=y)

	else:
		plt.scatter(X_new[0, :], X_new[1, :], c=y)

# ------------------------------------------------------------------------------

def main(samples=1500, seed=1234, out=False):
	X, y = get_data()

	XX, yy = random_samples(X, y, samples, seed=seed)

	# plt.figure(figsize=(6,4), dpi=300)
	w = 3
	f, axarr = plt.subplots(3, w)
	f.set_size_inches((15, 15), forward=True)
	f.set_tight_layout(True)
	f.set_dpi(150)

	# rbf kernel
	for i, sigma in enumerate((1., 5., 1e1, 30., 5e1, 75., 1e2, 1e3, 1e4)):
		k = kpca.RBF_Kernel(sigma)
		p = kpca.kPCA(k, 2)

		sub = axarr[i/w, i%w]
		sub.set_title(r"$\sigma = "+ str(sigma) + "$")
		X_new = p.fit_transform(XX)
		sub.scatter(X_new[0, :], X_new[1, :], c=yy)

		#plot_projected(p.fit_transform(XX), yy, sub=axarr[i / w, i % w])

	# polynomial kernel
	w2=3
	f2, axarr2 = plt.subplots(2,w2)
	f2.set_size_inches((15, 10), forward=True)
	f2.set_tight_layout(True)
	f2.set_dpi(150)

	for i, (d,c) in enumerate(((1,0),(1,1),(2,0),(2,-0.5),(2,1),(3,0))):
		k = kpca.Polynomial_Kernel(d)
		p = kpca.kPCA(k, 2)

		sub = axarr2[i/w2, i%w2]
		sub.set_title(r"degree of $"+ str(d) + "$, $c=" + str(c) + "$")
		X_new = p.fit_transform(XX)
		sub.scatter(X_new[0, :], X_new[1, :], c=yy)

	# 3d plot
	fig = plt.figure(figsize=(10, 3/4. * 10), dpi=150)
	ax = fig.add_subplot(111, projection='3d')

	k = kpca.RBF_Kernel(15.)
	p = kpca.kPCA(k, 3)
	X_new = p.fit_transform(XX)
	ax.scatter(X_new[0, :], X_new[1, :], X_new[2,:], c=yy)
	ax.set_title(r"RBF: $\sigma=15$")

	if out:
		f.savefig("rbf.pdf", dpi=400, format='pdf', bbox_inches='tight', frameon=False)
		f2.savefig("poly.pdf", dpi=400, format='pdf', bbox_inches='tight', frameon=False)
		fig.savefig("rbf3d.pdf", dpi=400, format='pdf', bbox_inches='tight', frameon=False)


	plt.show()


# ------------------------------------------------------------------------------

def do_pca(kernel, sigma, c, d, out, path=path, seed=1234, samples=1000):
	if kernel == "rbf":
		k = kpca.RBF_Kernel(sigma)
	elif kernel == "poly":
		if c is None:
			c = 0
		k = kpca.Polynomial_Kernel(d,c)
	elif kernel == "lin":
		k = kpca.Linear_Kernel()

	if seed is None:
		seed=1234
	if samples is None:
		samples=1000

	X, y = get_data(path = path)
	X_samples, y_samples = random_samples(X, y, samples, seed=seed)

	p = kpca.kPCA(k, 2)

	X_new = p.fit_transform(X_samples)
	plt.scatter(X_new[0, :], X_new[1, :], c=y_samples)

	if out:
		plt.savefig(out, dpi=600, format='png', bbox_inches='tight', frameon=False)

	plt.show()


# ------------------------------------------------------------------------------

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Execute code for demonstration (for correctors.) Calls without"
	                                 "parameters result in overview of different kernels used for kPCA.")
	parser.add_argument("-o", "--out", help="save plot as OUT", type=str)
	parser.add_argument("-k", "--kernel", help="choose kernel as rbf|lin|poly", type=str)
	parser.add_argument("-c", "--const", help="additive const of polynomial kernel", type=float)
	parser.add_argument("-s", "--sigma", help="rbf hyperparameter sigma", type=float)
	parser.add_argument("-d", "--degree", help="degree of polynomial kernel", type=int)
	parser.add_argument("--samples", help="number of random samples", type=int)
	parser.add_argument("--seed", help="seed for random()", type=int)

	args = parser.parse_args()

	# texify matplotlib graphics
	#Direct input
	plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
	#Options
	params = {
		'text.usetex' : True,
		'font.size' : 10,
		'font.family' : 'lmodern',
		'text.latex.unicode': True,
		'axes.labelsize': 10, # fontsize for x and y labels (was 10)
		'axes.titlesize': 14,
		'legend.fontsize': 8, # was 10
		'xtick.labelsize': 10,
		'ytick.labelsize': 10,
		'lines.linewidth': 0.4,
		'axes.linewidth': 0.6,
		'patch.linewidth': 0.4,
	}
	plt.rcParams.update(params)


	if args.kernel:
		do_pca(args.kernel, args.sigma, args.const, args.degree, args.out, args.samples, args.seed)
	else:
		out = True if args.out else False
		if args.seed is None and args.samples is None:
			main(out=out)
		elif args.samples:
			if args.seed:
				main(samples=args.samples, seed=args.seed, out=out)
			else:
				main(samples=args.samples, out=out)
		elif args.seed:
			main(seed=args.seed, out=out)
