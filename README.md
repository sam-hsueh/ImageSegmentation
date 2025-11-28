这是一个标注、训练、实例分割于一体的方案;

训练和预测部分是按https://github.com/IntptrMax/YoloSharp 的代码添加了UI;

UI框架是https://github.com/MaterialDesignInXAML/MaterialDesignInXamlToolkit;

下载https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.9.0%2Bcu130.zip文件
将上述文件解压后，将...libtorch-win-shared-with-deps-2.9.0+cu130\libtorch\lib文件夹下的DLL文件拷贝至...\bin\...runtimes\win-x64\native文件夹下;

下载https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth拷贝至SamModels文件夹下;

引用DLL目录下所有文件;

标注、训练、预测：
<img width="1920" height="1140" alt="屏幕截图 2025-11-12 120319" src="https://github.com/user-attachments/assets/9dac0810-bfcd-4cb7-9c99-ee7f94df7eb9" />

<img width="1920" height="1140" alt="屏幕截图 2025-11-28 234227" src="https://github.com/user-attachments/assets/65c9a070-e149-4a3c-a19b-d60aa50cf300" />

<img width="1920" height="1140" alt="屏幕截图 2025-11-12 185103" src="https://github.com/user-attachments/assets/a2f2b51b-b0c5-414c-a535-7c6633886a87" />

