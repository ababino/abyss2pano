# abyss2pano
Command line tool to stitch videos from the abyss rig.

## Install
You will need python2.

Install dependencies
```bash
pip install git+https://github.com/google/spatial-media.git
```

Install abyss2pano
```bash
pip install git+https://github.com/ababino/abyss2pano.git
```

## Usage

### Help:

```bash
abyss2pano.py -h

usage: abyss2pano.py [-h] [--campaign CAMPAIGN] [--date DATE] [--rig {1,2}]
                     [--out_frame_rate OUT_FRAME_RATE]
                     [--stitch {cube,sphere}] [--ext {MP4,LRV}]

Tool to stitch videos from the abyss rig. HERO 7 wide FOV: Vertical: 94.4,
Horizontal 122.6 Example: python abyss2pano.py --campaign Bimini-2019 --date
2019-08-24 --rig 1 --out_frame_rate 30 --stitch cube --ext LRV

optional arguments:
  -h, --help            show this help message and exit
  --campaign CAMPAIGN   path to campaign folder. Can be either relative or
                        absolute (e.g. Bimini-2019).
  --date DATE           date with the files to stitch in yyyy-mm-dd format
                        (e.g. 2019-08-26).
  --rig {1,2}           rig number (1 or 2)
  --out_frame_rate OUT_FRAME_RATE
                        frame rate of the output video
  --stitch {cube,sphere}
                        type of stitching. cube or sphere
  --ext {MP4,LRV}       extension of the files to stitch. The MP4 files are
                        the full resolution ones, and the LRV files are low
                        resolution. Use LRV for faster stitching.

```

### Example:

With this settings the command will produce a ~1GB file
```bash
abyss2pano.py --campaign Bimini-2019 --date 2019-08-24 --rig 1 --out_frame_rate 30 --stitch cube --ext LRV
```

With this settings the command will produce a ~7GB file
```bash
abyss2pano.py --campaign Bimini-2019 --date 2019-08-24 --rig 1 --out_frame_rate 30 --stitch cube --ext MP4
```
Changing the output frame rate to 120 produces ~8GB files, and it takes 4010s to process 60s.
