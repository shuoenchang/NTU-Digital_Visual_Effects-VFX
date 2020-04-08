# VFX Project 1 - High Dynamic Range Imaging

## 1. Team Members
* R08922A02 張碩恩
* R08922144 翁紫瑄

## 2. Environment
* ubuntu 18.04.4
* python==3.8.2
* numpy==1.18.1
* opencv==4.1.2
* exifread==2.1.2
* matplotlib==3.2.1

## 3. Usage

### 3.1 Data directory
```
[data]/
├──[dataset name]
├────[image1]
├────...
└────time.txt (option)
```

### 3.2 Input image spec
* If the input image include exif that included exposure time, then don't need time.txt file.
* Otherwise, you need to have a file time.txt that included exposure time for every picture.

Example for time.txt

```
0.03125
0.0625
0.125
0.25
0.5
1
2
4
8
16
32
64
128
256
512
1024
```

### 3.3 Run
```
python HDR.py --dataset [dataset name] --alpha [alpha]
```

* There are two optional argument
  * --ratio: Avoid so long computition time, it can reshape to the smaller image first.
  * --no-align: Do the reconstruct without alignment.

## 4. Result
* Input images: `data/waterfall`

### 4.1 Parameter
* --dataset waterfall
* --alpha 0.5
* --ratio 2
* --align

### 4.2 Result
![](result/waterfall_0.5.png)

