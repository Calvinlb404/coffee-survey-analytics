#!/usr/bin/env python3
# FTIR Data Analysis for Coffee Samples
# This script performs preprocessing and PCA on FTIR spectra of coffee samples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import os

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

def load_and_preprocess(filename, window_length=11, polyorder=2, deriv=1):
    """
    Load FTIR data from CSV and apply preprocessing
    
    Parameters:
    -----------
    filename : str
        Path to CSV file containing FTIR data
    window_length : int
        Window length for Savitzky-Golay filter
    polyorder : int
        Polynomial order for Savitzky-Golay filter
    deriv : int
        Derivative order (0 for smoothing, 1 for first derivative)
        
    Returns:
    --------
    wavenumbers : numpy.ndarray
        Array of wavenumbers
    spectra_processed : numpy.ndarray
        Preprocessed spectra data
    labels : list or None
        Sample labels (if available)
    """
    print(f"Loading data from {filename}...")
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Load data
    df = pd.read_csv(filename)
    
    # Print column names to help with debugging
    # print(f"Column names: {df.columns.tolist()}")
    
    # Separate wavenumber axis and spectral intensity data
    wavenumbers = df.iloc[:, 0].values             # First column = wavenumber
    spectra_data = df.iloc[:, 1:].values.T         # Transpose to get shape: n_samples x n_points
    
    # Extract sample names
    sample_names = df.columns[1:]
    print(df.columns)
    print(df.columns[1:])
    print(f"Found {len(sample_names)} samples")
    
    # Try to extract labels from sample names
    if any(['arabica' in name.lower() or 'robusta' in name.lower() for name in sample_names]):
        print("Extracting coffee type labels from sample names...")
        labels = []
        for name in sample_names:
            if 'arabica' in name.lower():
                labels.append('Arabica')
            elif 'robusta' in name.lower():
                labels.append('Robusta')
            else:
                labels.append('Unknown')
        print(f"Labels: {labels}")
    else:
        labels = None
        print("No coffee type labels found in sample names")
    
    # Apply Savitzky-Golay filter
    print(f"Applying Savitzky-Golay filter (window={window_length}, order={polyorder}, deriv={deriv})...")
    spectra_smoothed = savgol_filter(spectra_data, window_length=window_length, 
                                     polyorder=polyorder, deriv=deriv, axis=1)
    
    # Apply SNV normalization
    print("Applying Standard Normal Variate (SNV) normalization...")
    spectra_processed = apply_snv(spectra_smoothed)
    
    return wavenumbers, spectra_processed, labels

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
    
    return pca_scores, pca

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

    # File path
    data_file = './data/all_data.csv'
    
    # Parameters for preprocessing
    window_length = 11  # must be odd
    polyorder = 2
    deriv = 1  # 1 for first derivative, 0 for smoothing only
    
    try:
        # Load and preprocess data
        wavenumbers, spectra_processed, labels = load_and_preprocess(
            data_file, window_length, polyorder, deriv
        )
        
        # Perform PCA
        pca_scores, pca = perform_pca(spectra_processed,n_components=2)
        
        # Plot PCA results
        plot_pca_results(pca_scores, pca, labels)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 