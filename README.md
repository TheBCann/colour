# Color Grade Analysis

A Python-based tool for extracting and analyzing color grading information from professional film stills, with machine learning capabilities for film stock emulation.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/TheBCann/color-grade.git
cd color-grade

# Install dependencies
pip install -r requirements.txt

# Add your images to the pictures folder
cp your_image.jpg pictures/

# Run the analysis
python image_analysis_script.py

# Check results in the data folder
ls data/
```

## Overview

One of my past hobbies was competing in film festivals, where I primarily served as the cinematographer on set. I particularly enjoyed the color grading process and developed a deep appreciation for its artistic and technical aspects. Now as a computer science student, I've decided to combine these two areas of interest into this project.

This project analyzes high-quality stills from movies to extract color grading information, which can then be applied to your own creative projects. By dissecting the color data from professional films, users can learn from and implement industry-standard color grading techniques.

## Features

- **Comprehensive Color Analysis**
  - RGB channel statistics (mean, standard deviation, median)
  - Color histograms and cumulative distribution functions
  - Tonal range analysis (shadows, midtones, highlights)
  - Color ratio calculations across different tonal ranges
  - Tone mapping slope analysis
  - Shadow rolloff characterization

- **Machine Learning Film Emulation**
  - Neural network model for film stock emulation
  - Spectral response modeling based on real film characteristics
  - Support for different film stocks (e.g., Kodak Vision3 250D)

- **Visualization and Export**
  - Comprehensive analysis plots
  - JSON export of all color grading data
  - Side-by-side comparisons of original and analyzed images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TheBCann/color-grade.git
cd color-grade
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure data folder path (optional):
   - The default `DATA_FOLDER` is set to `/home/beans/Desktop/Projects/color-grade/data`
   - To use a relative path, edit the `DATA_FOLDER` variable in both files:
     ```python
     DATA_FOLDER = 'data'  # Creates data folder in current directory
     ```

### Dependencies
- pandas
- numpy
- Pillow
- seaborn
- opencv-python
- tensorflow
- IPython
- matplotlib
- scipy
- pyparsing (required by matplotlib)

## Usage

### Basic Image Analysis

1. Place your images in the `pictures` directory
2. Run the analysis script:
```bash
python image_analysis_script.py
```

The script will:
- Process all images in the `pictures` directory
- Generate comprehensive analysis plots
- Save results as JSON files in the `data` folder
- Create visualization PNG files for each analyzed image

### Interactive Analysis with Jupyter Notebook

For interactive exploration and visualization, use the included Jupyter notebook:

```bash
jupyter notebook color_session.ipynb
```

The notebook allows you to:
- Analyze individual images step by step
- Modify parameters and see results in real-time
- Experiment with different visualization styles
- Test new analysis methods before adding them to the main script

### Film Emulation Model

To use the machine learning film emulation model:
```python
from film_emulation_ml_model import create_film_emulation_model, FilmStock, SpectralResponse

# Define a film stock with spectral response curves
kodak_250d = FilmStock(
    name="Kodak Vision3 250D",
    iso=250,
    color_temp=5500,
    spectral_responses={
        'red': SpectralResponse([400, 500, 600, 700], [0.1, 0.3, 0.7, 0.9]),
        'green': SpectralResponse([400, 500, 600, 700], [0.2, 0.8, 0.6, 0.3]),
        'blue': SpectralResponse([400, 500, 600, 700], [0.9, 0.7, 0.2, 0.1])
    }
)

# Create the model
model = create_film_emulation_model((None, None, 3), kodak_250d)
```

### Working with Analysis Results

Load and use the analysis data in your own projects:

```python
import json
import numpy as np

# Load analysis results
with open('data/your_image_analysis.json', 'r') as f:
    analysis = json.load(f)

# Access color statistics
red_mean = analysis['color_stats']['red']['mean']
shadow_ratios = analysis['color_ratios']['shadows']

# Use tone mapping data for color grading
tone_curves = analysis['tone_mapping']
shadow_slope = tone_curves['red']['shadows']
```

## Technical Details

### Color Analysis Methodology

The analysis script performs several sophisticated color grading analyses:

1. **Color Statistics**: Calculates mean, standard deviation, and median for each RGB channel
2. **Tonal Range Analysis**: Segments the image into shadows (0-84), midtones (85-170), and highlights (171-255)
3. **Color Ratios**: Computes R/G, R/B, and G/B ratios for each tonal range
4. **Tone Mapping**: Analyzes the slope of cumulative distribution functions
5. **Shadow Rolloff**: Calculates second derivatives to characterize shadow behavior

### Understanding the Metrics

- **Color Statistics**: Reveal overall color cast and contrast characteristics
- **Tonal Range Analysis**: Shows how colors are distributed across brightness levels
- **Color Ratios**: Indicate color grading choices (e.g., warm shadows, cool highlights)
- **Tone Mapping Slopes**: Describe contrast and dynamic range compression
- **Shadow Rolloff**: Measures how smoothly shadows transition to black (film-like vs digital)

These metrics can help you:
- Reverse-engineer professional color grades
- Match the look of reference images
- Understand why certain images have specific moods
- Create consistent color grades across projects

### Output Format

Analysis results are saved in JSON format with the following structure:
```json
{
  "color_stats": {
    "red": {"mean": 0.0, "std": 0.0, "median": 0.0},
    "green": {"mean": 0.0, "std": 0.0, "median": 0.0},
    "blue": {"mean": 0.0, "std": 0.0, "median": 0.0}
  },
  "tonal_ranges": {...},
  "color_ratios": {...},
  "tone_mapping": {...},
  "shadow_rolloff": {...}
}
```

## Project Structure

```
color-grade/
├── image_analysis_script.py      # Main analysis script
├── film-emulation-ml-model.py    # ML model for film emulation
├── color_session.ipynb           # Jupyter notebook for experimentation
├── colorscience.pdf              # Reference material on color science
├── requirements.txt              # Python dependencies
├── test.py                       # Simple matplotlib test script
├── pictures/                     # Input images directory
│   ├── goldenHour*.jpg          # Golden hour lighting samples
│   ├── inside*.jpg              # Interior lighting samples
│   ├── gritty*.jpg              # Stylized/graded samples
│   └── [your images here]
├── data/                        # Output directory (created automatically)
│   ├── *_analysis.json          # Analysis data
│   └── *_analysis.png           # Visualization plots
└── README.md                    # This file
```

## Example Output

The analysis generates comprehensive visualizations including:

- **Original Image Display**: Shows the input image being analyzed
- **Color Histograms**: RGB channel distribution across the image
- **Color Curves (CDF)**: Cumulative distribution functions for each channel
- **Color Statistics Table**: Mean, standard deviation, and median values
- **Color Ratios Chart**: R/G, R/B, and G/B ratios across tonal ranges
- **Tone Mapping Slopes**: Response curves for shadows, midtones, and highlights
- **Tonal Range Analysis**: Detailed breakdown with percentage coverage
- **Shadow Rolloff Values**: Mathematical characterization of shadow behavior

## Troubleshooting

### ModuleNotFoundError: No module named 'pyparsing'
This error can occur when matplotlib dependencies are not fully installed. Fix with:
```bash
pip install pyparsing
# or reinstall matplotlib
pip install --upgrade --force-reinstall matplotlib
```

### Memory Issues with Large Images
For very high-resolution images, you may encounter memory errors. Consider:
- Resizing images before processing
- Processing images in batches
- Increasing system swap space

### Color Profile Warnings
If you see "Could not convert color profile" warnings, the script will continue with the original color space. For best results, use sRGB images.

## Sample Images

The `pictures` directory includes various test images demonstrating different lighting conditions:
- Golden hour shots (warm tones, high dynamic range)
- Interior scenes (mixed lighting, shadow detail)
- Night photography (low light, color cast analysis)
- Close-up portraits (skin tone analysis)
- Gritty/stylized shots (heavy color grading)

## Contributing

This is a work in progress and is actively being developed. Any contributions would be greatly appreciated!

### Areas for Contribution
- **Film Stock Profiles**: Add spectral response data for popular film stocks
- **Color Science**: Implement additional color space transformations
- **Performance**: Optimize numpy operations and add GPU support
- **Documentation**: Add tutorials and example workflows
- **Testing**: Create unit tests for analysis functions
- **GUI Development**: Build a user-friendly interface with PyQt or Tkinter
- **Algorithm Enhancement**: Improve shadow rolloff detection and tone mapping
- **Integration**: Add support for popular editing software (DaVinci, Premiere)

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Development

- **Enhanced Color Space Support**: Add LAB, HSV, and XYZ color space analysis
- **LUT Generation**: Export color grades as 3D LUTs for use in video editing software
- **Film Stock Library**: Build a comprehensive database of film stock characteristics
- **GPU Acceleration**: Implement CUDA/OpenCL support for faster processing
- **Web Interface**: Create a Flask/Django web app for browser-based analysis
- **Video Support**: Extend analysis to video files with temporal color analysis
- **Machine Learning Enhancement**: Train models on professional colorist data
- **Batch Processing**: Add multi-threading for processing large image sets
- **RAW Support**: Integrate with rawpy for camera RAW file processing
- **Color Match Tool**: Automatically match color grades between images

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by professional color grading tools like DaVinci Resolve and FilmConvert
- Special thanks to the cinematography community for sharing knowledge about film characteristics

---

For questions, suggestions, or collaboration opportunities, please open an issue or submit a pull request.
