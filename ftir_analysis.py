#!/usr/bin/env python3
# FTIR Data Analysis for Coffee Samples
# This script performs preprocessing and PCA on FTIR spectra of coffee samples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import os
from scipy.integrate import cumtrapz

def apply_snv(spectra_matrix):
    """
    Apply Standard Normal Variate (SNV): for each sample (row), 
    subtract mean and divide by std.
    """
    mean = spectra_matrix.mean(axis=1, keepdims=True)
    std = spectra_matrix.std(axis=1, keepdims=True)
    return (spectra_matrix - mean) / std

def combine_data(filenames):
    """
    Combine file data into one table, with labels
    """
    new_df = None
    for file in filenames:
        full_path = f'./data/{file}'
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
    
        # Load data
        if new_df is None:
            new_df = pd.read_csv(full_path)
        else:
            to_add = pd.read_csv(full_path)
            new_df[file[:-4]] = to_add.iloc[:, 1].values
    new_df.to_csv('all_data.csv', index=False)

def transpose_data(file):
    """
    Transpose csv data, and label columns by index
    """
    full_path = f'./data/{file}'
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    df = pd.read_csv(full_path)
    df_transposed = df.transpose()
    df_transposed.columns = [f'Taste_{num}' for num in range(1,81)]
    df_transposed.to_csv('transposed_tasting.csv', index=False)

def load_and_preprocess(filename, window_length=11, polyorder=2, deriv=1, type='ftir'):
    """
    Load FTIR data from CSV and apply preprocessing. Returns everything needed for inversion.
    """
    print(f"Loading data from {filename}...")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    df = pd.read_csv(filename)

    labels = None
    wavenumbers = None
    if type == 'ftir':
        wavenumbers = df.iloc[:, 0].values
        spectra_data = df.iloc[:, 1:].values.T
        sample_names = df.columns[1:]
        labels = [''.join([name[0], name[4], name[8]]) for name in sample_names]
    elif type == 'tasting':
        spectra_data = df.iloc[:, :15].values.T
        sample_names = df.columns[:15]
        labels = [name[-2:] for name in sample_names]

    print(f"Labels: {labels}")
    print("spectra_data shape:", spectra_data.shape)
    print("Number of labels:", len(labels))

    # Apply Savitzky-Golay filter
    print(f"Applying Savitzky-Golay filter (window={window_length}, order={polyorder}, deriv={deriv})...")
    spectra_smoothed = savgol_filter(spectra_data, window_length=window_length, 
                                     polyorder=polyorder, deriv=deriv, axis=1)

    # Capture the initial value (first point) of each filtered spectrum (needed to invert derivative)
    initial_values = spectra_smoothed[:, 0]  # shape: (n_samples,)

    # Compute mean and std of each spectrum before SNV normalization
    means = spectra_smoothed.mean(axis=1)  # shape: (n_samples,)
    stds = spectra_smoothed.std(axis=1)

    # Apply SNV normalization
    print("Applying Standard Normal Variate (SNV) normalization...")
    spectra_processed = (spectra_smoothed - means[:, None]) / stds[:, None]

    return wavenumbers, spectra_processed, spectra_data, labels, initial_values, means, stds

def invert_snv(snv_data, original_means, original_stds):
    # snv_data: (n_samples, n_features)
    # original_means, original_stds: (n_samples,)
    return snv_data * original_stds[:, None] + original_means[:, None]

def invert_savgol_derivative(deriv_signal, dx=1.0, initial_value=0.0):
    # Integrate along axis=1 (features)
    # Returns same shape if we append the initial value manually
    integrated = cumtrapz(deriv_signal, dx=dx, axis=1, initial=0.0)
    return integrated + initial_value  # You need to supply this per sample

def invert_processing(spectra_processed, original_means, original_stds, initial_values,
                      window_length=11, polyorder=2, deriv=1, dx=1.0):
    print("Inverting SNV normalization...")
    snv_inverted = invert_snv(spectra_processed, original_means, original_stds)

    print("Inverting Savitzky-Golay derivative...")
    spectra_reconstructed = invert_savgol_derivative(snv_inverted, dx=dx, initial_value=initial_values)

    return spectra_reconstructed

def perform_pca(spectra_processed, n_components=2):
    """
    Perform Principal Component Analysis
    
    Parameters:
    -----------
    spectra_processed : numpy.ndarray
        Preprocessed spectra data
    n_components : int
        Number of principal components to extract
        
    Returns:
    --------
    pca_scores : numpy.ndarray
        PCA scores (coordinates in PC space)
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    """
    print(f"Performing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(spectra_processed)
    
    # Calculate explained variance
    var_exp = pca.explained_variance_ratio_ * 100
    for i, v in enumerate(var_exp):
        print(f"PC{i+1} explained variance: {v:.2f}%")
    print(np.shape(pca.components_))
    
    return pca_scores, pca

def reconstruct_from_PCA(z, pca):
    # z is a (n_components,) vector in PCA space (e.g., one row of X_transformed)
    # For example: z = [k, 0, 0, ..., 0] to activate only PC1
    """
    Reconstruct a vector from PCA space using a fitted sklearn PCA object.
    """
    return pca.mean_ + z @ pca.components_

def plot_graph(x,y,title):
    """
    Plot x and y, spectra FTIR
    """

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, color='teal')
    
    plt.xlabel("Wavenumber", fontsize=12)
    plt.ylabel("Spectra", fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

def plot_pca_results(pca_scores, pca, labels=None, title="PCA of Coffee FTIR Spectra"):
    """
    Visualize PCA results
    
    Parameters:
    -----------
    pca_scores : numpy.ndarray
        PCA scores (coordinates in PC space)
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    labels : list or None
        Sample labels for coloring
    title : str
        Plot title
    """
    var_exp = pca.explained_variance_ratio_ * 100
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        labels = np.array(labels)
        for lab in np.unique(labels):
            idx = np.where(labels == lab)
            plt.scatter(pca_scores[idx, 0], pca_scores[idx, 1], label=str(lab), alpha=0.8, s=100)
    else:
        plt.scatter(pca_scores[:, 0], pca_scores[:, 1], color='teal', alpha=0.8, s=100)
    # Annotate axes with percentage of variance explained
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=12)
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=12)
    plt.title(title, fontsize=14)
    if labels is not None:
        plt.legend(title="Coffee Type", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('pca_plot.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # all_files = ['DarkFastFast.csv','DarkSlowFast.csv','DarkFastSlow.csv','DarkSlowSlow.csv',
    #              'LightFastFast.csv','LightSlowFast.csv','LightFastSlow.csv','LightSlowSlow.csv']
    # combine_data(all_files)
    
    #transpose_data('Tasting.csv')

    # File path
    data_file = './data/all_ftir.csv'
    
    # Parameters for preprocessing
    window_length = 11  # must be odd
    polyorder = 2
    deriv = 1  # 1 for first derivative, 0 for smoothing only
    
    try:
        # Load and preprocess data
        wavenumbers, spectra_processed, og_spectra, labels,initial_values, means, stds = load_and_preprocess(
            data_file, window_length, polyorder, deriv, type='ftir'
        )
        
        #-----------------
        # Perform PCA
        N = 5
        pca_scores, pca = perform_pca(spectra_processed,n_components=N)
        print(pca_scores)
        
        # Plot PCA results
        plot_pca_results(pca_scores, pca, labels)
        
        print("Analysis completed successfully!")
        #-----------------

        #----Reconstruct PC1
        z = np.zeros(N,1)
        z[0] = 1
        pc1 = reconstruct_from_PCA(z, pca)

        #invert smoothing, filter
        pc1_reconstructed = invert_processing(pc1,means,stds,initial_values)
        plot_graph(wavenumbers, og_spectra[1,:], "Sample 1")
        plot_graph(wavenumbers, pc1_reconstructed[1,:], "PC1")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()