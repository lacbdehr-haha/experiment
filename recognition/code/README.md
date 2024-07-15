# **A Low-Quality 3D Face Recognition Method based on Contrastive Learning ** 

The implementation code for this method is open source here. Due to high storage consumption of the dataset, only partial dataset examples are uploaded. Some implementations are still being optimized and updated in real time. If needed, please contact via email. My email address is [zhangn202203@163.com](mailto:zhangn202203@163.com).

The method is applicable for low-quality 3D face data recognition. In this experiment, normal maps and depth maps of low-quality facial images were chosen to validate the effectiveness and real-time performance of the proposed method. The recognition accuracy for two different types of images significantly outperforms current state-of-the-art methods. 

The code consists of functional modules located in several folders: `data`, `lock3dface`, `model`, `original`, `Pointflowmodel`, and `scu`. Here is a brief introduction to each module:

1. data: Stores the training and testing datasets processed from the lock3dface and Extend-multi-dim datasets.
2. lock3dface: The implementation process for recognizing facial normal maps from the lock3dface dataset.
3. model: Store model training parameters for testing purposes.
4. original: Baseline experiment.
5. Pointflowmodel: Continuous Normalizing Flow (CNF) implementation.
6. scu: The implementation process for recognizing facial depth maps from the Extend-multi-dim dataset.
7. objectives.py: Implementation of the comparative loss function.

The training and testing files for each dataset have filenames that include 'train' and 'test' as part of their names. Explanation of relevant module functionalities covered in the .py files under each folder, not detailed here one by one.