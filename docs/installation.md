# Installation Guide

Data Preprocessing Toolkit

This document explains how to install and use the Data Preprocessing Toolkit
in different scenarios.

---

## Option 1 — Download via GitHub Release (Recommended for Beginners)

This is the easiest way to use the toolkit without Git or pip.

### Steps
- Open the GitHub repository
- Go to the **Releases** tab
- Download the file
- Extract the ZIP file
- Copy the folder `data_preprocessing_toolkit/` into your project directory
- Install dependencies:

    ```
    pip install -r requirements.txt
    ```

- Import to your python file:

    ```python
    from data_preprocessing_toolkit.pipeline import cleaning_pipeline
    ```

