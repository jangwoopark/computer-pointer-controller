# Computer Pointer Controller

*TODO:* Write a short introduction to your project

The Computer Pointer Controller is an application that uses a gaze detection model to control the mouse pointer using an input video or a live stream from a webcam.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

###Environment

Download miniconda: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
Use python 3.6 (not 2.7 or 3.8, Ubuntu 20.04 Focal Fossa is not compatible with this code)

Install: bash Miniconda3-latest-Linux-x86_64.sh

Create environmet like conda create --name pointer-controller

Activate it: conda activate pointer-controller

Change to starter folder and install requirements to conda: 

conda install python=3.6
pip install -r requirements.txt
Also pip install opencv-python PyAutoGUI (do not install opencv directly from conda)

Ubuntu 20.04 focal fossa installation of openvino through apt: intel-openvino-runtime-ubuntu18-2020.3.194 and intel-openvino-dev-ubuntu18-2020.3.194. This version is best compatible with python=3.6; lower or higher versions have bugs.
Source openvino: source /opt/intel/openvino/bin/setupvars.sh

###Models to download

python3.6 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-0001 --precisions FP32,FP16,INT8

python3.6 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP32,FP16,INT8

python3.6 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP32,FP16,INT8

python3.6 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP32,FP16,INT8

All four models will be downloaded into the intel folder which can be moved to the starter folder (see directory structure)

###Directory Structure using du -a starter/ in linux command line

16	starter/.Instructions.md.swp
8	starter/README.md
2328	starter/bin/demo.mp4
0	starter/bin/.gitkeep
2332	starter/bin
4	starter/requirements.txt
232	starter/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
4120	starter/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.bin
4356	starter/intel/face-detection-adas-0001/FP32
232	starter/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
2060	starter/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin
2296	starter/intel/face-detection-adas-0001/FP16
6656	starter/intel/face-detection-adas-0001
7472	starter/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin
52	starter/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml
7528	starter/intel/head-pose-estimation-adas-0001/FP32
3736	starter/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.bin
52	starter/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
3792	starter/intel/head-pose-estimation-adas-0001/FP16
11324	starter/intel/head-pose-estimation-adas-0001
7356	starter/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.bin
64	starter/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml
7424	starter/intel/gaze-estimation-adas-0002/FP32
3680	starter/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.bin
64	starter/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml
3748	starter/intel/gaze-estimation-adas-0002/FP16
11176	starter/intel/gaze-estimation-adas-0002
44	starter/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml
748	starter/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin
796	starter/intel/landmarks-regression-retail-0009/FP32
44	starter/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml
376	starter/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin
424	starter/intel/landmarks-regression-retail-0009/FP16
1224	starter/intel/landmarks-regression-retail-0009
30384	starter/intel
4	starter/src/model.py
8	starter/src/facial_landmarks_detection.py
4	starter/src/__pycache__/gaze_estimation.cpython-36.pyc
4	starter/src/__pycache__/head_pose_estimation.cpython-36.pyc
8	starter/src/__pycache__/facial_landmarks_detection.cpython-37.pyc
4	starter/src/__pycache__/gaze_estimation.cpython-37.pyc
4	starter/src/__pycache__/face_detection.cpython-38.pyc
4	starter/src/__pycache__/input_feeder.cpython-38.pyc
8	starter/src/__pycache__/facial_landmarks_detection.cpython-36.pyc
4	starter/src/__pycache__/mouse_controller.cpython-37.pyc
4	starter/src/__pycache__/head_pose_estimation.cpython-37.pyc
4	starter/src/__pycache__/mouse_controller.cpython-38.pyc
4	starter/src/__pycache__/mouse_controller.cpython-36.pyc
4	starter/src/__pycache__/face_detection.cpython-37.pyc
4	starter/src/__pycache__/input_feeder.cpython-37.pyc
4	starter/src/__pycache__/face_detection.cpython-36.pyc
4	starter/src/__pycache__/input_feeder.cpython-36.pyc
72	starter/src/__pycache__
4	starter/src/face_detection.py
4	starter/src/mouse_controller.py
12	starter/src/main.py
8	starter/src/gaze_estimation.py
4	starter/src/input_feeder.py
4	starter/src/head_pose_estimation.py
124	starter/src
32872	starter/


## Demo
*TODO:* Explain how to run a basic demo of your model.

python src/main.py -face_m intel/face-detection-adas-0001/FP32/face-detection-adas-0001 -head_pose_m intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -facial_m intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -gaze_m intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i bin/demo.mp4 -d CPU -pt 0.6 -flags face_detect face_landmark_detect head_pose gaze_est

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

main.py has the following arguments: 

-h: Get information about all the command line arguments
-face_m: (required) Specify the path of Face Detection model's name as  shown below for specific precision FP32, FP16, INT8
-facial_m: (required) Specify the path of Facial landmarks Detection model's name for specific precision FP32, FP16
-head_pose_m: (required) Specify the path of hose pose Detection model's name for specific precision FP32, FP16
-gaze_m: (required) Specify the path of gaze estimation model's name for specific precision FP32, FP16, INT8
-i: (required) Specify the path of input video file or enter cam for taking input video from webcam.
-l: (optional) Specify the absolute path of cpu extension if some layers of models are not supported on the device.  
-d: (optional) Specify the target device to infer the video file on the model. Supported devices are: CPU, GPU, FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD. 
-pt: (optional) Specify the probability threshold for face detection model to detect the face accurately from video frame.
-flag: (required) Specify the flags from face_detect, face_landmark_detect, head_pose, gaze_est to visualize the output of corresponding models of each frame seperated by space.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

My Ubuntu operating system is installed in an external hard drive allowing me to use it by booting it in different machines like my desktop and laptop whenever convenient. The same OS will be used in two different hardwares for benchmarking this project.

My current Intel Xeon CPU desktop can show specifics by running cat /proc/cpuinfo in the linux command line:

processor	: 0 to 7
vendor_id	: GenuineIntel
cpu family	: 6
model		: 30
model name	: Intel(R) Xeon(R) CPU           X3480  @ 3.07GHz
stepping	: 5
microcode	: 0xa
cpu MHz		: 1259.242
cache size	: 8192 KB
physical id	: 0
siblings	: 8
core id		: 0
cpu cores	: 4
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 11
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm sse4_1 sse4_2 popcnt lahf_lm pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid dtherm flush_l1d
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips	: 6118.10
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

My current Intel i7 CPU laptop can show specifics by running cat /proc/cpuinfo in the linux command line:

processor	: 0 to 3
vendor_id	: GenuineIntel
cpu family	: 6
model		: 42
model name	: Intel(R) Core(TM) i7-2640M CPU @ 2.80GHz
stepping	: 7
microcode	: 0x2f
cpu MHz		: 1949.158
cache size	: 4096 KB
physical id	: 0
siblings	: 4
core id		: 0
cpu cores	: 2
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx lahf_lm epb pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm ida arat pln pts md_clear flush_l1d
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips	: 5581.87
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:


Precision  Type of Hardware	Total inference time	Total load time	fps

FP32		desktop cpu		82.01 s	0.84 s			0.7194244604316546
FP16		desktop cpu		81.72 s	0.93 s			0.7219774840920216

FP32		laptop cpu		89.69 s	13.82 s		0.6578213847697625
FP16		laptop cpu		88.59 s	1.41 s			0.6659893893215938

FP32 desktop cpu:
Face detection load time in seconds: 330.45 ms
Facial Landmark detection load time in seconds: 433.06 ms
Head pose detection load time in seconds: 557.76 ms
Gaze estimation load time in seconds: 700.20 ms

FP16 desktp cpu:
Face detection load time in seconds: 346.15 ms
Facial Landmark detection load time in seconds: 453.43 ms
Head pose detection load time in seconds: 616.90 ms
Gaze estimation load time in seconds: 820.71 ms

FP32 laptop cpu:
Face detection load time in seconds: 2194.10 ms
Facial Landmark detection load time in seconds: 2316.98 ms
Head pose detection load time in seconds: 2482.22 ms
Gaze estimation load time in seconds: 2594.63 ms

FP16 laptop cpu:
Face detection load time in seconds: 314.92 ms
Facial Landmark detection load time in seconds: 402.35 ms
Head pose detection load time in seconds: 516.74 ms
Gaze estimation load time in seconds: 653.13 ms

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

The best inference time was the FP16 desktop cpu while the best loading time was the FP32 also desktop cpu. The best fps was the FP32 laptop cpu, but notice the huge load time for it. The laptop has a little higher inference time due to the weaker processing power compared to the desktop. However, the FP16 performed a little less but still relatively well in load time and fps compared to the desktop. For edge devices that need to be mobile FP16 would be a great choice given the processing power in a laptop. Actually, the loading time for each part in the FP16 laptop surpasses the loading time in the FP32 and FP16 desktops as seen above.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

I would have to say that not many users have an OS installed in an external hard drive to be used through an enclosure connected to an USB port. This gives me the chance to compare the benchmark for a stationary edge device like an Intel CPU desktop versus a mobile edge device like an Intel CPU laptop. 

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

When performing inference using the demo video as input, the mouse controller crashed when pointer moved to a corner of the screen. To overcome this problem, pyautogui.FailSafe is set to false in the MouseController class. This feature is enabled by default so that it can easily stop the execution of the pyautogui program by manually moving the mouse to the upper left corner of the screen.
