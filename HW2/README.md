# VFX Project 2 - Image Stitching

## 1. Team Members
* R08922A02 張碩恩
* R08922144 翁紫瑄

## 2. Environment
* ubuntu 18.04.4
* python==3.8.2
* numpy==1.18.1
* opencv==4.1.2
* matplotlib==3.2.1
* scipy==1.4.1

## 3. Usage

### 3.1 Data directory
```
[data]/
├──[dataset name]
├─────[image1]
├─────...
└─────pano.txt (required)
```

### 3.2 Input image spec
* `pano.txt` is a file that included focal length for each picture that estimated by [autostich](http://matthewalunbrown.com/autostitch/autostitch.html)
* For autostitch, you need to use the old 32-bit Windows version
* The file looks like the following:

```
D:\courses\vfx\2005Spring\projects\proj#2\test_data\parrington\prtn00.jpg
384 512

1 0 255.5 
0 1 191.5 
0 0 1 

0.999837 0.0125485 0.0129957 
-0.0125474 0.999921 -0.000163239 
-0.0129968 1.77461e-008 0.999916 

704.916

```

where 704.916 is the estimated focal length for the first image.


### 3.3 Run
```
python main.py --dataset [dataset name] (--left or --right)
```

* Argument description
  * --dataset [dataset name] : Name of dataset.
  * --ratio [ratio] : Avoid so long computition time, it can reshape to the smaller image first.
  * --align : Do end2end alignment after stitch.
  * --left : First picture is in left.
  * --right : First picture is in right.

## 4. Result
* Input images: `data/parrington`

### 4.1 Parameter
* --dataset parrington
* --ratio 1
* --align
* --right

### 4.2 Result
![](result/parrington_crop.png)

