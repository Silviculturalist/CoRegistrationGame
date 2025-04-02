# Co-Registration Game

The **Co-Registration Game** is an interactive tool designed for forest resource management. It allows users to align (or “co-register”) tree plot data with canopy height model (CHM), or other saved data using an intuitive graphical interface. The application supports manual adjustments via a range of keyboard shortcuts and provides options for data imputation based on height-diameter relationships.

## Table of Contents

- [Overview](#overview)
- [Input Data Requirements](#input-data-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Outputs](#outputs)
- [License and Credits](#license-and-credits)
- [Citations](#citations)

## Overview

The Co-Registration Game helps forest managers and researchers to:
- **Visualize** tree and CHM data in a unified viewport.
- **Adjust** tree plot positions interactively using rotation, translation, and flipping.
- **Optimize** the registration between field-measured tree data and remotely sensed CHM data.
- **Impute** missing tree parameters (e.g., height or diameter) using a Näslund (1936) height-diameter relationship.

The tool integrates a Tkinter-based startup menu for data selection and configuration with a Pygame-based interactive display for the co-registration process.

## Input Data Requirements

The program requires two CSV files:

1. **Tree Data File** (Field Data):  
   Expected columns (default names):
   - `Stand` (or mapped to a different column via startup options)
   - `PLOT`
   - `TreeID`
   - `X_GROUND`
   - `Y_GROUND`
   - `STEMDIAM` (in centimeters; will be converted to meters internally)
   - Optionally, `Species`, `XC`, and `YC` for plot center information

2. **CHM Data File** (Canopy Height Model):  
   Expected columns (default names):
   - `X`
   - `Y`
   - `H` (height in a specified unit; conversion is applied based on user settings)
   - `IDALS` (unique tree identifier)

**Note:**  
The startup menu allows you to map your CSV column names to the required fields. If your data file does not include a stand identifier column (or any other column), you can leave the mapping empty; in that case, all rows will be assumed to belong to the provided stand ID.

## Installation

### Using Conda

A sample `environment.yml` is provided to create a Python virtual environment with the required dependencies.

```yaml
name: CoRegGameEnv
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - scipy
  - matplotlib
  - pygame
  - pynput
  - tk
```

To create and activate the environment: 
```bash
conda env create -f environment.yml
conda activate CoRegGameEnv
```

### Using pip
Alternatively, install the dependencies via pip: 
```bash
pip install numpy pandas scipy matplotlib pygame pynput tk
```

## Usage
Launch the startup menu by running: 
```bash
python startup.py
```

The startup menu (built with Tkinter) will allow you to:

- Select the Tree Data File and CHM Data File.
- Specify CSV separators.
- Map your CSV column names to the required fields (StandID, PlotID, TreeID, X, Y, DBH, H).
- Choose whether to impute missing values for DBH or Height using the provided Näslund height-diameter relationship (only one imputation option per file is allowed).
- Set Näslund model parameters (with a real-time preview of the height curve).
- Select an output folder where the transformed tree data and transformation logs will be saved.

Once configured, click `Start` to launch the interactive application.

## Keyboard Shortcuts
During the interactive co-registration session, the following shortcuts are available:

### Viewport Navigation:

`W`, `A`, `S`, `D`: Pan the viewport up, left, down, and right.

### Plot Adjustment:

Arrow Keys (`Up`, `Down`, `Left`, `Right`): Shift the current plot.

`E`: Rotate the current plot counterclockwise.

`R`: Rotate the current plot clockwise.

`F`: Flip the current plot vertically.

`1`, `2`: Zoom in and out.

`6`, `7`: Increase or decrease the tree display scale.

`8`: Reset the tree display scale to default.

### Plot Management:

`J`: Join – attempts to compute an optimal plot alignment using a Fractional ICP algorithm.

`C`: Confirm – saves the current plot’s transformation.

`N`: Skip the current plot.

Period (`.`): Mark the current plot as unplaceable (do not save its position).

`B`: Step back – revert the last confirmed plot and restore its previous state.

### Other:

`Space`: Toggle flash mode (visualize different data layers).

## Outputs
The program generates the following outputs:

### Transformed Tree Data:
After confirming plots, the transformed tree positions (including computed transformations such as rotation, translation, and flipping) are saved as CSV files in the ./Trees directory (or the output folder you specify).

### Transformation Logs:
Detailed logs of the transformations applied to each plot are saved in the ./Transformations directory.

## Graphical Outputs:
The interactive display shows both the tree data and CHM data aligned together, with plot centers and other diagnostic overlays.


## License and Credits
Co-Registration Game was developed by Carl Vigren at Dept. for Forest Resource Management at the Swedish University of Agricultural Sciences.
For more details and contributions, please see [https://github.com/Silviculturalist/CoRegistrationGame].


## Citations
The software has been used in the following published works. If used further, please do provide proper acknowledgement by referring to one of the following works. 

Holmgren J., Vigren C. 2025. *Estimation of tree stem attributes using samples of mobile laser scanning combined with complete coverage of airborne laser scanning*.

Vigren, C. 2024. *Pushing the Envelope: Empirical Growth Models for Forests at Change*. Ph.D. Diss. Swedish University of Agricultural Sciences. 2024:97. pages 119-122. pp. 156. DOI: https://doi.org/10.54612/a.7qt3hgmn6k. 
(In Print Edition Only: Manuscript IV Appendix 1.)
