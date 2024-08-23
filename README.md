# 3D Scanner Laser Triangulation

## Environment setup

Run these commands on terminal for MacOS or Command Prompt in Windows

Clone the git project

```
git clone https://github.com/Anugrah-T-Sebastian/3D_Scanner_Laser_Triangulation.git
```

Navigate to the cloned project

```
cd 3D_Scanner_Laser_Triangulation
```

Create virtual environment

```
python -m venv env
```

Activate Virtual Environment
For MacOS

```
source env/bin/activate  # On Unix or MacOS
```

or
For Windows

```
env/Scripts/activate  # On Windows
```

Install Libraries from requirements.txt:

```
pip install -r requirements.txt
```

## Camera calbiration Data preparation

Navigate to `Calibration/Chessboard ` and print out the _camera_calibration1.pdf_ file. Take various snapshots of the Chessboard on the print bed in different orientation using the camera of the printer head attachments. Store the snapshots in the `Calibration` folder.

## Scanning Data processing

1. For data captured through video, put them in `Video_sample` and for data captured through Snapshots measured steps, put them in the `Measured_steps_Data`.

2. Now open up the `Experiments.ipynb` jupyter file and the run cells as needed to see the results of the experiements.

3. The intermediate results such are edge trace and depth maps are stored in `edge_trace` and `depth_maps` folders.
