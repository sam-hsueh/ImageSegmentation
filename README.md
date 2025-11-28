这是一个标注、训练、实例分割于一体的方案;

训练和预测部分是按https://github.com/IntptrMax/YoloSharp 的代码添加了UI;

UI框架是https://github.com/MaterialDesignInXAML/MaterialDesignInXamlToolkit;

下载https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.9.0%2Bcu130.zip文件
将上述文件解压后，将...libtorch-win-shared-with-deps-2.9.0+cu130\libtorch\lib文件夹下的DLL文件拷贝至...\bin\...runtimes\win-x64\native文件夹下;

下载https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth拷贝至SamModels文件夹下;

引用DLL目录下所有文件;

标注、训练、预测：
![屏幕截图 2025-05-21 231146](https://github.com/user-attachments/assets/df3ba4ea-9cd3-4d5d-a9df-a6d099d03bb7)
![屏幕截图 2025-05-21 231248](https://github.com/user-attachments/assets/4d9a11c6-7709-41aa-b108-342a0fc9cf41)
![屏幕截图 2025-05-21 230422](https://github.com/user-attachments/assets/0e34ecb6-3b5a-46f8-9c98-5ae272881f0a)
